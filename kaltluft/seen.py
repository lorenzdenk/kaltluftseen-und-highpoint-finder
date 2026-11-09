"""Kaltluftseen aus dem Strömungsmodell: Gebiete, in denen die Kaltluft dick liegt und kaum fliesst."""
import numpy as np
from scipy import ndimage

# ESA-WorldCover-Klassen
LC_SIEDLUNG = 50
LC_WASSER = 80


def anteile(lc_fein, veg_fein, faktor, veg_min=3.0):
    """Flächenanteile je Modellzelle aus den feinen Rastern: Wald (Vegetation >= veg_min), mittlere Bestandeshöhe,
    Siedlung und Wasser (aus der Landbedeckung, falls vorhanden)."""
    nr, nc = veg_fein.shape[0] // faktor, veg_fein.shape[1] // faktor

    def block(a):
        return a[:nr * faktor, :nc * faktor].reshape(nr, faktor, nc, faktor).mean(axis=(1, 3), dtype=np.float32)

    ist_wald = veg_fein >= veg_min
    wald = block(ist_wald)
    wald_h = block(np.where(ist_wald, veg_fein, 0)) / np.maximum(wald, 1e-6)
    if lc_fein is None:
        return wald, wald_h, np.zeros_like(wald), np.zeros_like(wald)
    return wald, wald_h, block(lc_fein == LC_SIEDLUNG), block(lc_fein == LC_WASSER)


def finde_stauseen(zustand, z, res, min_h=5.0, max_tempo=0.2, min_flaeche_ha=1.0, wasser=None):
    """Zusammenhängende Gebiete mit Schichtdicke >= min_h und Fliessgeschwindigkeit <= max_tempo.

    Rückgabe: (Liste von dicts, Label-Raster int32 mit Feld 'lab', -1 sonst).
    """
    h = zustand["h"]
    v = zustand["tempo"]
    m = np.isfinite(h) & (h >= min_h) & (v <= max_tempo)
    if wasser is not None:
        m &= wasser < 0.5  # offene Seeflächen sind kein Kaltluftsee
    m = ndimage.binary_opening(m, structure=np.ones((3, 3), bool))
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    lab = lab.astype(np.int32) - 1
    if n == 0:
        return [], lab
    idx = np.arange(n)
    cell = res * res
    cnt = np.bincount(lab[m], minlength=n)
    vol = np.bincount(lab[m], weights=h[m], minlength=n) * cell
    vsum = np.bincount(lab[m], weights=v[m], minlength=n)
    hmax = ndimage.maximum(h, lab, idx)
    zmin_pos = ndimage.minimum_position(np.where(np.isfinite(z), z, np.inf), lab, idx)
    out = []
    for p in np.nonzero(cnt * cell / 1e4 >= min_flaeche_ha)[0]:
        r, c = zmin_pos[p]
        out.append(dict(typ="Stau", lab=int(p), r=int(r), c=int(c), volumen=float(vol[p]),
                        flaeche_ha=float(cnt[p] * cell / 1e4), tiefe_max=float(hmax[p]),
                        tiefe_mittel=float(vol[p] / (cnt[p] * cell)), tempo_mittel=float(vsum[p] / cnt[p]),
                        boden=float(z[r, c]), spiegel=float(z[r, c] + h[r, c])))
    return out, lab


def richtung(qx, qy):
    """Fliessrichtung als Kompasskurs, wohin die Kaltluft fliesst (Grad, 0 = Norden, 90 = Osten). y zeigt nach Süden."""
    return (np.degrees(np.arctan2(qx, -qy)) % 360).astype(np.float32)
