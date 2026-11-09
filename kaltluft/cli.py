"""Kommandozeile: `python -m kaltluft daten ...` lädt Eingangsdaten, `python -m kaltluft rechnen ...` sucht Kaltluftseen."""
import argparse
import glob
import json
import os
import time

import numpy as np
from shapely.geometry import Point, box
from shapely.prepared import prep

from . import ausgabe, daten, seen, senken, stroemung
from .raster import Gitter, block_mittel, geom_to_wgs84, lies_mosaik, log, to_wgs84, umrisse


def parser():
    ap = argparse.ArgumentParser(prog="kaltluft", description="Kaltluftseen aus swissALTI3D: Senken und Kaltluftabfluss.")
    sub = ap.add_subparsers(dest="befehl", required=True)

    d = sub.add_parser("daten", help="Eingangsdaten laden (nur was im Cache fehlt)")
    r = sub.add_parser("rechnen", help="Senken suchen, Kaltluftabfluss simulieren, Ergebnisse schreiben")
    for p in (d, r):
        p.add_argument("--name", default="thurgau", help="Name des Gebiets")
        p.add_argument("--cache", default="cache")
        p.add_argument("--rand-km", type=float, default=2, help="Rand um das Gebiet (Kaltluft fliesst von aussen zu)")
        p.add_argument("--fenster", help="nur Ausschnitt: x0,y0,x1,y1 in LV95 (m)")
    d.add_argument("--canton", type=int, default=20, help="Kantonsnummer (BFS), 0 = nur --bbox")
    d.add_argument("--bbox", default="8.65,47.36,9.52,47.71", help="W,S,E,N in Grad, falls keine Kantonsgrenze")
    d.add_argument("--threads", type=int, default=8)
    d.add_argument("--ohne-vegetation", action="store_true")
    d.add_argument("--ohne-landbedeckung", action="store_true")
    d.add_argument("--ohne-karte", action="store_true", help="keine Strassen und Ortsnamen für die Übersichtskarten")

    r.add_argument("--out", default="out")
    r.add_argument("--out-name", help="Name der Ausgabedateien (Standard: --name)")
    r.add_argument("--alles", action="store_true", help="nicht auf die Gebietsgrenze beschränken")
    r.add_argument("--top", type=int, default=200, help="zusätzliche GPX-Datei mit den N grössten je Art")
    g = r.add_argument_group("Senken (feines Raster)")
    g.add_argument("--res", type=float, default=5, help="Raster der Senkenanalyse (m)")
    g.add_argument("--veg-k", type=float, default=0.4, help="wirksamer Anteil der Vegetationshöhe als Damm (0 = aus)")
    g.add_argument("--veg-min", type=float, default=3, help="Vegetation ab dieser Höhe (m) zählt als Wald/Hecke")
    g.add_argument("--veg-max", type=float, default=12, help="höchste wirksame Dammhöhe durch Vegetation (m)")
    g.add_argument("--min-tiefe", type=float, default=1.0, help="Senke: Mindesttiefe (m)")
    g.add_argument("--min-flaeche", type=float, default=0.2, help="Senke: Mindestfläche offenes Gelände (ha)")
    g.add_argument("--min-volumen", type=float, default=1000, help="Senke: Mindestvolumen (m³)")
    g = r.add_argument_group("Kaltluftabfluss (grobes Raster)")
    g.add_argument("--ohne-stroemung", action="store_true", help="nur Senken rechnen")
    g.add_argument("--res-fluss", type=float, default=20, help="Raster des Strömungsmodells (m), Vielfaches von --res")
    g.add_argument("--stunden", default="1,2,4", help="Auswertezeiten nach Beginn der Abkühlung (h)")
    g.add_argument("--see-stunde", type=float, help="Zeitpunkt für die GPX-Ausgabe (Standard: mittlerer Wert)")
    g.add_argument("--see-schicht", type=float, default=5, help="Kaltluftsee: Schichtdicke mindestens (m)")
    g.add_argument("--see-tempo", type=float, default=0.2, help="Kaltluftsee: Fliessgeschwindigkeit höchstens (m/s)")
    g.add_argument("--see-flaeche", type=float, default=1.0, help="Kaltluftsee: Mindestfläche (ha)")
    P = stroemung.Parameter
    for k, hilfe in (("dT", "Temperaturdefizit der Kaltluft (K)"),
                     ("prod_frei", "Kaltluftproduktion Freiland (m³/m²h)"), ("prod_wald", "Produktion Wald"),
                     ("prod_siedlung", "Produktion Siedlung"), ("cd_grund", "Widerstandsbeiwert Freiland"),
                     ("cd_wald", "zusätzlicher Widerstand Wald"), ("cd_siedlung", "zusätzlicher Widerstand Siedlung")):
        g.add_argument("--" + k.replace("_", "-"), type=float, default=getattr(P, k), help=f"{hilfe} (Standard %(default)s)")
    return ap


def _fenster(a):
    return tuple(map(float, a.fenster.split(","))) if a.fenster else None


def rechne(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    name = a.out_name or a.name
    f = int(round(a.res_fluss / a.res))
    if abs(f * a.res - a.res_fluss) > 1e-6 or f < 1:
        raise SystemExit("--res-fluss muss ein Vielfaches von --res sein")

    dtm = sorted(glob.glob(os.path.join(a.cache, "dtm2_*.tif")))
    if not dtm:
        raise SystemExit("keine Höhenmodell-Kacheln im Cache: zuerst `python -m kaltluft daten` laufen lassen")
    gpfad = os.path.join(a.cache, f"boundary_{a.name}.geojson")
    grenze = daten.lies_grenze(gpfad) if os.path.exists(gpfad) else None
    fen = _fenster(a)
    if fen:
        gebiet = box(*fen)
    elif grenze is not None and not a.alles:
        gebiet = grenze
    else:
        import rasterio
        bs = [rasterio.open(p).bounds for p in dtm]
        gebiet = box(min(b.left for b in bs), min(b.bottom for b in bs), max(b.right for b in bs), max(b.top for b in bs))
        a.rand_km = 0
    x0, y0, x1, y1 = gebiet.bounds
    m = a.rand_km * 1000
    grob = Gitter((x0 - m, y0 - m, x1 + m, y1 + m), a.res_fluss)
    fein = Gitter((grob.x0, grob.y0, grob.x1, grob.y1), a.res)
    drin = prep(gebiet)

    log(f"Höhenmodell lesen ({len(dtm)} Kacheln), Raster {a.res:g} m: {fein.nc} x {fein.nr} Zellen")
    Z = lies_mosaik(dtm, fein, "average", nodata=-9999)
    if not np.isfinite(Z).any():
        raise SystemExit("keine Höhendaten im Gebiet")
    vegf = sorted(glob.glob(os.path.join(a.cache, "veg_*.tif")))
    if vegf:
        log(f"Vegetationshöhe lesen ({len(vegf)} Kacheln)")
        V = lies_mosaik(vegf, fein, "average", nodata=255) / 2.0
        V[~np.isfinite(V)] = 0
    else:
        log("keine Vegetationshöhe im Cache: nur Gelände")
        V = np.zeros_like(Z)

    log("Senken suchen")
    sen, lab_s = senken.finde_senken(Z, V, a.res, a.veg_k, a.veg_min, a.veg_max, a.min_tiefe, a.min_flaeche,
                                     a.min_volumen)
    for s in sen:
        s["X"], s["Y"] = fein.xy(s["r"], s["c"])
        ux, uy = fein.xy(s["ueberlauf_r"], s["ueberlauf_c"])
        s["ueberlauf_lon"], s["ueberlauf_lat"] = to_wgs84(ux, uy)
        s["fuellzeit_h"] = (s["volumen"] / (s["einzugsgebiet_ha"] * 1e4 * a.prod_frei)
                            if s["einzugsgebiet_ha"] > 0 else None)
    sen = [s for s in sen if drin.contains(Point(s["X"], s["Y"]))]
    log(f"  {len(sen)} Senken: " + ", ".join(f"{k} {sum(s['typ'] == k for s in sen)}" for k in ("Senke", "Vegetation")))
    geo = umrisse(lab_s, [s["lab"] for s in sen], fein)
    for s in sen:
        s["geom"] = geo.get(s["lab"])
    del lab_s

    stau_alle = []
    haupt = None
    if not a.ohne_stroemung:
        stunden = sorted(float(x) for x in a.stunden.split(","))
        haupt = a.see_stunde if a.see_stunde is not None else stunden[len(stunden) // 2]
        if haupt not in stunden:
            raise SystemExit("--see-stunde muss in --stunden vorkommen")
        lcf = sorted(glob.glob(os.path.join(a.cache, "lc_*.tif")))
        LC = None
        if lcf:
            log(f"Landbedeckung lesen ({len(lcf)} Kacheln)")
            LC = np.nan_to_num(lies_mosaik(lcf, fein, "nearest", nodata=0)).astype(np.uint8)
        else:
            log("keine Landbedeckung im Cache: Siedlung und Wasser werden wie Freiland behandelt")
        zc = block_mittel(Z, f)
        wald, wald_h, siedlung, wasser = seen.anteile(LC, V, f, a.veg_min)
        del LC
        par = stroemung.Parameter(**{k: getattr(a, k) for k in ("dT", "prod_frei", "prod_wald", "prod_siedlung",
                                                               "cd_grund", "cd_wald", "cd_siedlung")})
        prod, cd0, cdo, hob = stroemung.felder(par, np.isfinite(zc), wald, wald_h, siedlung, wasser)
        log(f"Kaltluftabfluss simulieren: Raster {a.res_fluss:g} m, {grob.nc} x {grob.nr} Zellen, bis {stunden[-1]:g} h")
        sim = stroemung.simuliere(zc, a.res_fluss, stunden, par, prod, cd0, cdo, hob, log=log)
        bz = sim["bilanz"]
        log(f"  erzeugt {bz['erzeugt_m3'] / 1e9:.2f} km³ Kaltluft, davon noch {bz['vorhanden_m3'] / 1e9:.2f} km³ im Gebiet")
        ausgabe.schreibe_raster(a.out, name, grob, sim)
        # Orientierung für die Karten: Strassen (10-m-Raster) und Ortsnamen, soweit im Cache
        sf = sorted(glob.glob(os.path.join(a.cache, "strassen_*.tif")))
        gk = Gitter((grob.x0, grob.y0, grob.x1, grob.y1), min(10.0, a.res_fluss))
        strassen = np.nan_to_num(lies_mosaik(sf, gk, "max", nodata=0)).astype(np.uint8) if sf else None
        orte = [o for o in daten.lies_orte(a.cache) if grob.x0 < o[1] < grob.x1 and grob.y0 < o[2] < grob.y1]
        for st, zu in sim["zustand"].items():
            objs, lab = seen.finde_stauseen(zu, zc, a.res_fluss, a.see_schicht, a.see_tempo, a.see_flaeche,
                                            wasser if lcf else None)
            for o in objs:
                o["X"], o["Y"] = grob.xy(o["r"], o["c"])
                o["nach_h"] = st
            objs = [o for o in objs if drin.contains(Point(o["X"], o["Y"]))]
            geo = umrisse(lab, [o["lab"] for o in objs], grob, vereinfachen=a.res_fluss * 0.5)
            for o in objs:
                o["geom"] = geo.get(o["lab"])
            log(f"  nach {st:g} h: {len(objs)} Kaltluftseen (Stau), "
                f"{sum(o['flaeche_ha'] for o in objs):.0f} ha")
            stau_alle += objs
            maske = np.isin(lab, [o["lab"] for o in objs])
            ausgabe.schreibe_karte(os.path.join(a.out, f"kaltluft_{name}_{st:g}h.png"), grob, zc, zu, maske,
                                   [(s["X"], s["Y"]) for s in sen],
                                   f"Kaltluft {name} nach {st:g} h: Schichtdicke, Fliessrichtung, Kaltluftseen",
                                   strassen, orte)
        # Strömung am Ort jeder Senke und jedes Stausees
        zu = sim["zustand"][haupt]
        for o in sen + stau_alle:
            r, c = grob.rc(o["X"], o["Y"])
            if 0 <= r < grob.nr and 0 <= c < grob.nc and np.isfinite(zu["h"][r, c]):
                zo = sim["zustand"][o["nach_h"]] if "nach_h" in o else zu
                o.update(schicht=float(zo["h"][r, c]), tempo=float(zo["tempo"][r, c]),
                         strom=float(np.hypot(zo["qx"][r, c], zo["qy"][r, c])),
                         richtung=float(seen.richtung(zo["qx"][r, c], zo["qy"][r, c])),
                         beginn=float(sim["beginn"][r, c]))

    objekte = _beschreibe(sen, stau_alle, haupt, a)
    ausgabe.schreibe_objekte(a.out, name, objekte, haupt, a.top)
    with open(os.path.join(a.out, f"parameter_{name}.json"), "w") as fh:
        json.dump({k: v for k, v in vars(a).items()}, fh, indent=1)
    log(f"fertig in {(time.time() - t0) / 60:.1f} min")
    return objekte


def _beschreibe(sen, stau, haupt, a):
    """Rang, Name, Beschreibung und WGS84-Geometrie je Objekt. Rang je Art nach Volumen."""
    sen.sort(key=lambda o: -o["volumen"])
    stau.sort(key=lambda o: (o["nach_h"] != haupt, o["nach_h"], -o["volumen"]))
    for k, o in enumerate(sen, 1):
        o["rang"] = k
        grund = ("geschlossene Mulde" if o["typ"] == "Senke"
                 else f"Abfluss durch Wald/Hecke gestaut, Bäume heben den Spiegel um {o['stau_baeume']:.1f} m")
        o["name"] = f"KLS {k} {ausgabe.fmt_vol(o['volumen'])}"
        t = (f"{o['typ']}: {grund}. Volumen {ausgabe.fmt_vol(o['volumen'])}, Fläche {o['flaeche_ha']:.1f} ha, "
             f"Tiefe max. {o['tiefe_max']:.1f} m / Ø {o['tiefe_mittel']:.1f} m, Spiegel {o['spiegel']:.0f} m ü. M., "
             f"Einzugsgebiet {o['einzugsgebiet_ha']:.1f} ha")
        if o.get("fuellzeit_h") is not None:
            t += f" (gefüllt nach ca. {o['fuellzeit_h'] * 60:.0f} min)"
        if o.get("schicht") is not None:
            t += (f". Strömungsmodell nach {haupt:g} h: Kaltluftschicht {o['schicht']:.0f} m, "
                  f"{o['tempo']:.1f} m/s Richtung {o['richtung']:.0f}°")
        o["beschreibung"] = t + f". Überlauf bei {o['ueberlauf_lat']:.5f}, {o['ueberlauf_lon']:.5f}"
    rang = {}
    for o in stau:
        rang[o["nach_h"]] = o["rang"] = rang.get(o["nach_h"], 0) + 1
        o["name"] = f"Stau {o['rang']} {o['flaeche_ha']:.0f} ha {o['tiefe_max']:.0f} m ({o['nach_h']:g} h)"
        o["beschreibung"] = (f"Kaltluftstau nach {o['nach_h']:g} h: Schicht mind. {a.see_schicht:g} m dick und langsamer "
                             f"als {a.see_tempo:g} m/s. Fläche {o['flaeche_ha']:.1f} ha, Schichtdicke max. "
                             f"{o['tiefe_max']:.0f} m / Ø {o['tiefe_mittel']:.0f} m, Volumen "
                             f"{ausgabe.fmt_vol(o['volumen'])}, mittlere Geschwindigkeit {o['tempo_mittel']:.2f} m/s. "
                             f"Punkt = tiefste Stelle ({o['boden']:.0f} m ü. M.)")
    objekte = sen + stau
    for k, o in enumerate(objekte, 1):
        o["id"] = k
        o["lon"], o["lat"] = to_wgs84(o["X"], o["Y"])
        o["geom_wgs84"] = geom_to_wgs84(o["geom"]) if o.get("geom") is not None else None
    return objekte


def main(argv=None):
    a = parser().parse_args(argv)
    if a.befehl == "daten":
        fehler = daten.lade(a.name, a.cache, a.canton, a.bbox, a.rand_km, a.threads, not a.ohne_vegetation,
                            not a.ohne_landbedeckung, _fenster(a), not a.ohne_karte)
        raise SystemExit(1 if fehler else 0)
    rechne(a)
