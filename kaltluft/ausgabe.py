"""Ausgabe: GPX (Wegpunkte, Umrisse), GeoJSON, CSV, GeoTIFF und Übersichtskarten (PNG)."""
import csv
import json
import os
from xml.sax.saxutils import escape

import numpy as np
from shapely.geometry import mapping

from .raster import log, schreibe_tif
from .seen import richtung

QUELLE = ("Datenquelle: swissALTI3D © swisstopo; Vegetationshöhenmodell LiDAR LFI, BAFU/WSL; ESA WorldCover 2021. "
          "Modellrechnung, nicht gemessen.")

SPALTEN = [("id", "id"), ("typ", "typ"), ("nach_h", "nach_h"), ("volumen", "volumen_m3"), ("flaeche_ha", "flaeche_ha"),
           ("flaeche_offen_ha", "flaeche_offen_ha"), ("tiefe_max", "tiefe_max_m"), ("tiefe_mittel", "tiefe_mittel_m"),
           ("spiegel", "spiegel_m"), ("boden", "boden_tiefster_punkt_m"), ("lat", "lat"), ("lon", "lon"),
           ("ueberlauf_lat", "ueberlauf_lat"), ("ueberlauf_lon", "ueberlauf_lon"), ("stau_baeume", "stau_durch_baeume_m"),
           ("einzugsgebiet_ha", "einzugsgebiet_ha"), ("fuellzeit_h", "fuellzeit_h"),
           ("schicht", "kaltluftschicht_m"), ("tempo", "fliessgeschwindigkeit_m_s"), ("strom", "volumenstrom_m2_s"),
           ("richtung", "fliessrichtung_grad"), ("beginn", "schicht_5m_nach_h"), ("tempo_mittel", "tempo_mittel_m_s")]


def fmt_vol(v):
    if v >= 1e6:
        return f"{v / 1e6:.1f} Mio m³"
    if v >= 1e4:
        return f"{v / 1e3:.0f}'000 m³"
    return f"{v:.0f} m³"


def _rund(k, v):
    if isinstance(v, float):
        if v != v:
            return None
        return round(v, 6 if k.endswith(("lat", "lon")) else 2)
    return v


def _ringe(geom):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    return [p.exterior.coords for p in polys if not p.is_empty]


def schreibe_objekte(out, name, objekte, haupt_h, top=200):
    """objekte: Liste von dicts (bereits sortiert, mit id, name, beschreibung, lat, lon, geom_wgs84 ...)."""
    base = os.path.join(out, f"kaltluftseen_{name}")
    kopf = ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="kaltluftseen-finder" '
            'xmlns="http://www.topografix.com/GPX/1/1">\n<metadata><name>Kaltluftseen %s</name><desc>%s</desc>'
            '</metadata>\n' % (escape(name.capitalize()), escape(QUELLE)))
    gpx = [o for o in objekte if o["typ"] != "Stau" or o["nach_h"] == haupt_h]
    beste = [o for o in gpx if o["rang"] <= top]
    for pfad, sel in ((base + ".gpx", gpx), (base + f"_top{top}.gpx", beste)):
        with open(pfad, "w", encoding="utf-8") as f:
            f.write(kopf)
            for o in sel:
                f.write(f'<wpt lat="{o["lat"]:.6f}" lon="{o["lon"]:.6f}"><ele>{o["boden"]:.1f}</ele>'
                        f'<name>{escape(o["name"])}</name><desc>{escape(o["beschreibung"])}</desc>'
                        f'<type>{o["typ"]}</type></wpt>\n')
            f.write("</gpx>\n")
    with open(base + "_umrisse.gpx", "w", encoding="utf-8") as f:
        f.write(kopf)
        for o in gpx:
            if o.get("geom_wgs84") is None:
                continue
            f.write(f'<trk><name>{escape(o["name"])}</name><desc>{escape(o["beschreibung"])}</desc>'
                    f'<type>{o["typ"]}</type>')
            for ring in _ringe(o["geom_wgs84"]):
                f.write("<trkseg>" + "".join(f'<trkpt lat="{y:.6f}" lon="{x:.6f}"/>' for x, y in ring) + "</trkseg>")
            f.write("</trk>\n")
        f.write("</gpx>\n")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([s[1] for s in SPALTEN])
        for o in objekte:
            w.writerow(["" if o.get(k) is None else _rund(k, o[k]) for k, _ in SPALTEN])
    feats = []
    for o in objekte:
        if o.get("geom_wgs84") is None:
            continue
        props = {s: _rund(k, o[k]) for k, s in SPALTEN if o.get(k) is not None}
        props.update(name=o["name"], beschreibung=o["beschreibung"])
        feats.append({"type": "Feature", "properties": props, "geometry": mapping(o["geom_wgs84"])})
    with open(base + ".geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    log("geschrieben:", ", ".join(os.path.basename(base) + s
                                  for s in (".gpx", f"_top{top}.gpx", "_umrisse.gpx", ".csv", ".geojson")))


def schreibe_raster(out, name, gitter, sim):
    """Je Auswertezeit ein GeoTIFF mit Schichtdicke, Fliessgeschwindigkeit, Volumenstrom und Fliessrichtung."""
    namen = ["Schichtdicke (m)", "Fliessgeschwindigkeit (m/s)", "Volumenstrom je m Breite (m2/s)",
             "Fliessrichtung wohin (Grad, 0 = Nord)"]
    for st, zu in sim["zustand"].items():
        strom = np.hypot(zu["qx"], zu["qy"]).astype(np.float32)
        schreibe_tif(os.path.join(out, f"kaltluft_{name}_{st:g}h.tif"), gitter,
                     [zu["h"], zu["tempo"], strom, richtung(zu["qx"], zu["qy"])], namen)
    schreibe_tif(os.path.join(out, f"kaltluft_{name}_beginn.tif"), gitter, [sim["beginn"]],
                 ["Stunden, bis die Kaltluftschicht 5 m dick ist"])


def schattierung(z, res):
    zz = np.where(np.isfinite(z), z, np.nanmin(z))
    gy, gx = np.gradient(zz, res)
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy) * 2)
    aspect = np.arctan2(gy, -gx)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - np.pi / 2 - aspect), 0, 1)


def _beschrifte(ax, orte, ext, breite_zoll):
    """Ortsnamen setzen, grosse Orte zuerst; ein Name entfällt, wenn er einen schon gesetzten überdecken würde."""
    import matplotlib.patheffects as pe
    m_je_pt = (ext[1] - ext[0]) / (breite_zoll * 72.0)
    groesse = {0: 4.6, 1: 5.2, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.5}
    gesetzt = []
    rand = [pe.withStroke(linewidth=1.6, foreground="white", alpha=0.85)]
    for name, e, n, kl in orte:
        fs = groesse.get(kl, 5)
        w = 0.58 * fs * len(name) * m_je_pt
        h = 1.25 * fs * m_je_pt
        x0, y0 = e - w / 2, n + 0.25 * h
        if x0 < ext[0] or x0 + w > ext[1] or y0 < ext[2] or y0 + h > ext[3]:
            continue
        if any(x0 < b[2] and x0 + w > b[0] and y0 < b[3] and y0 + h > b[1] for b in gesetzt):
            continue
        gesetzt.append((x0, y0, x0 + w, y0 + h))
        ax.plot([e], [n], "s", ms=1.6 + 0.4 * kl, color="#1b1b1b", mew=0, zorder=6)
        ax.text(e, y0, name, fontsize=fs, ha="center", va="bottom", color="#111111", zorder=7,
                fontweight="bold" if kl >= 4 else "normal", path_effects=rand)
    return len(gesetzt)


def schreibe_karte(pfad, gitter, z, zu, stau_maske, senken_xy, titel, strassen=None, orte=None):
    """Übersichtskarte: Relief, Schichtdicke (ein Farbton, hell -> dunkel), Fliessrichtung, Umriss der Kaltluftseen,
    dazu Haupt- und Nebenstrassen (Raster mit Klassen 1..3) und Ortsnamen zur Orientierung."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False
    nr, nc = z.shape
    ext = (gitter.x0, gitter.x1, gitter.y0, gitter.y1)
    b = 18.0
    fig, ax = plt.subplots(figsize=(b, max(b * nr / nc, 3) + 1.0), dpi=200)
    ax.imshow(schattierung(z, gitter.res), cmap="gray", vmin=-0.2, vmax=1.1, extent=ext, interpolation="bilinear")
    h = zu["h"]
    blau = LinearSegmentedColormap.from_list("schicht", ["#dbe9f6", "#6aaed6", "#2171b5", "#08306b"])
    hm = float(np.nanpercentile(h, 99)) if np.isfinite(h).any() else 1.0
    im = ax.imshow(np.ma.masked_where(~(h >= 1.0), h), cmap=blau, vmin=0, vmax=max(hm, 5), alpha=0.72, extent=ext,
                   interpolation="nearest")
    if strassen is not None and strassen.any():
        rgba = np.zeros(strassen.shape + (4,), np.float32)
        rgba[strassen == 1] = (0.35, 0.33, 0.30, 0.30)   # Nebenstrassen: zurückhaltend
        rgba[strassen == 2] = (0.45, 0.20, 0.05, 0.85)   # Hauptstrassen
        rgba[strassen == 3] = (0.60, 0.25, 0.00, 1.00)   # Autobahn
        ax.imshow(rgba, extent=ext, interpolation="bilinear", zorder=3)
        ax.plot([], [], color=(0.45, 0.20, 0.05), lw=1.6, label="Haupt- und Nebenstrassen (swissTLM3D)")
    k = max(int(round(max(nr, nc) / 90)), 1)
    r0 = k // 2
    qx = zu["qx"][r0::k, r0::k]
    qy = zu["qy"][r0::k, r0::k]
    s = np.hypot(qx, qy)
    ok = np.isfinite(s) & (s > 0.05)
    if ok.any():
        xs = gitter.x0 + (np.arange(r0, nc, k) + 0.5) * gitter.res
        ys = gitter.y1 - (np.arange(r0, nr, k) + 0.5) * gitter.res
        X, Y = np.meshgrid(xs, ys)
        lg = np.sqrt(s[ok] / np.nanpercentile(s[ok], 95)).clip(0.15, 1.3)
        ax.quiver(X[ok], Y[ok], qx[ok] / s[ok] * lg, -qy[ok] / s[ok] * lg, color="#1b1b1b", scale=1.0 / (k * gitter.res),
                  scale_units="xy", width=0.0009, headwidth=4, alpha=0.75, zorder=4)
    if stau_maske is not None and stau_maske.any():
        ax.contour(np.linspace(ext[0], ext[1], nc), np.linspace(ext[3], ext[2], nr), stau_maske.astype(float),
                   levels=[0.5], colors="#d4145a", linewidths=1.3, zorder=5)
        ax.plot([], [], color="#d4145a", lw=1.3, label="Kaltluftsee (Stau): dick und kaum Bewegung")
    if senken_xy:
        ax.plot([p[0] for p in senken_xy], [p[1] for p in senken_xy], "o", ms=3.0, mfc="#ffd400", mec="#1b1b1b",
                mew=0.6, ls="", label="Senke im Gelände", zorder=5)
    ax.plot([], [], color="#1b1b1b", marker=">", ls="-", lw=0.8, ms=4, label="Fliessrichtung (Länge ~ Volumenstrom)")
    if orte:
        _beschrifte(ax, orte, ext, b * 0.9)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92).set_zorder(10)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title(titel, fontsize=12, loc="left")
    ax.set_xlabel("LV95 Ost (m)", fontsize=8)
    ax.set_ylabel("LV95 Nord (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style="plain", useOffset=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Dicke der Kaltluftschicht (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(pfad, bbox_inches="tight")
    plt.close(fig)
    return True
