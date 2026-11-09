"""Senkenanalyse: geschlossene Mulden im Gelände und Stau hinter Wald/Hecken (Priority-Flood, Barnes et al. 2014).

Eine Senke ist ein Bereich, aus dem Kaltluft nur über den tiefsten Randpunkt (Überlauf) abfliessen kann. Vegetation
wird als teildurchlässiger Damm behandelt: wirksame Oberfläche = Gelände + min(k * Vegetationshöhe, max).
"""
import numpy as np
from numba import njit

N8R = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
N8C = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)


@njit(cache=True)
def _hpush(hk, hv, n, k, v):
    i = n
    hk[i] = k
    hv[i] = v
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return n + 1


@njit(cache=True)
def _hpop(hk, hv, n):
    v = hv[0]
    n -= 1
    hk[0] = hk[n]
    hv[0] = hv[n]
    i = 0
    while True:
        l = 2 * i + 1
        if l >= n:
            break
        m = l
        if l + 1 < n and hk[l + 1] < hk[l]:
            m = l + 1
        if hk[i] <= hk[m]:
            break
        hk[m], hk[i] = hk[i], hk[m]
        hv[m], hv[i] = hv[i], hv[m]
        i = m
    return v, n


@njit(cache=True)
def priority_flood(Wf, valid, nr, nc):
    """Füllt alle abflusslosen Bereiche bis zum Überlauf (Arrays flach, Länge nr*nc).

    Rückgabe: F (gefüllte Höhe), src (Index der Überlaufzelle eines gefüllten Bereichs, sonst -1), parent
    (Abflussrichtung: Zelle, von der aus die Zelle erreicht wurde; -1 = Rand), order (Reihenfolge von unten nach oben).
    """
    N = nr * nc
    F = Wf.copy()
    src = np.full(N, -1, np.int32)
    parent = np.full(N, -1, np.int32)
    order = np.empty(N, np.int32)
    closed = ~valid
    hk = np.empty(N, np.float32)
    hv = np.empty(N, np.int32)
    pit = np.empty(N, np.int32)
    hn = 0
    ph = 0
    pt = 0
    no = 0
    for i in range(N):
        if not valid[i]:
            continue
        r = i // nc
        c = i - r * nc
        edge = r == 0 or c == 0 or r == nr - 1 or c == nc - 1
        if not edge:
            for k in range(8):
                if not valid[(r + N8R[k]) * nc + c + N8C[k]]:
                    edge = True
                    break
        if edge:
            closed[i] = True
            hn = _hpush(hk, hv, hn, F[i], i)
    while hn > 0 or ph < pt:
        if ph < pt:
            i = pit[ph]
            ph += 1
        else:
            i, hn = _hpop(hk, hv, hn)
        order[no] = i
        no += 1
        r = i // nc
        c = i - r * nc
        root = src[i] if src[i] >= 0 else i
        for k in range(8):
            rr = r + N8R[k]
            cc = c + N8C[k]
            if rr < 0 or cc < 0 or rr >= nr or cc >= nc:
                continue
            j = rr * nc + cc
            if closed[j]:
                continue
            closed[j] = True
            parent[j] = i
            if F[j] <= F[i]:
                F[j] = F[i]
                src[j] = root
                pit[pt] = j
                pt += 1
            else:
                hn = _hpush(hk, hv, hn, F[j], j)
    return F, src, parent, order[:no]


@njit(cache=True)
def _einzugsgebiete(lab, parent, order, npool):
    """Kaltluft-Einzugsgebiet je Senke: alle Zellen, deren Fliessweg zuerst in diese Senke führt (Anzahl Zellen)."""
    c = np.full(lab.size, -1, np.int32)
    cnt = np.zeros(npool, np.int64)
    for k in range(order.size):
        i = order[k]
        if lab[i] >= 0:
            c[i] = lab[i]
        else:
            p = parent[i]
            if p >= 0:
                c[i] = c[p]
        if c[i] >= 0:
            cnt[c[i]] += 1
    return cnt


@njit(cache=True)
def _statistik(lab, F, Z, V, F0, npool, vmin):
    cnt = np.zeros(npool, np.int64)
    cnt_offen = np.zeros(npool, np.int64)
    vol = np.zeros(npool, np.float64)
    dmax = np.zeros(npool, np.float32)
    imax = np.full(npool, -1, np.int64)
    lift = np.zeros(npool, np.float32)
    for i in range(lab.size):
        p = lab[i]
        if p < 0:
            continue
        d = F[i] - Z[i]
        if d < 0:
            d = 0.0
        cnt[p] += 1
        if V[i] < vmin:
            cnt_offen[p] += 1
        vol[p] += d
        if d > dmax[p]:
            dmax[p] = d
            imax[p] = i
        if F[i] - F0[i] > lift[p]:
            lift[p] = F[i] - F0[i]
    return cnt, cnt_offen, vol, dmax, imax, lift


def finde_senken(Z, V, res, veg_k=0.4, veg_min=3.0, veg_max=12.0, min_tiefe=1.0, min_flaeche_ha=0.2,
                 min_volumen=1000.0):
    """Sucht Senken (und Vegetationsstau, falls veg_k > 0) im Raster Z (2D, NaN = keine Daten), V = Vegetationshöhe.

    Rückgabe: (Liste von dicts je Senke, Label-Raster int32 mit dem Feld 'lab' der dicts, -1 sonst).
    """
    nr, nc = Z.shape
    cell = res * res
    Zf = Z.ravel()
    Vf = np.ascontiguousarray(V.ravel(), np.float32)
    valid = np.isfinite(Zf)
    Z0 = np.where(valid, Zf, np.nan).astype(np.float32)
    damm = np.where(Vf >= veg_min, np.minimum(Vf * veg_k, veg_max), 0).astype(np.float32) if veg_k > 0 else None
    E = Z0 if damm is None else (Z0 + damm).astype(np.float32)

    F, src, parent, order = priority_flood(E, valid, nr, nc)
    gefuellt = valid & (F - E > 0.05) & (src >= 0)
    roots, inv = np.unique(src[gefuellt], return_inverse=True)
    lab = np.full(Zf.size, -1, np.int32)
    lab[np.nonzero(gefuellt)[0]] = inv.astype(np.int32)
    ezg = _einzugsgebiete(lab, parent, order, roots.size)
    del parent, order, src
    # Vegetationsstau: Seespiegel läge ohne Bäume deutlich tiefer
    F0 = F if damm is None else priority_flood(Z0, valid, nr, nc)[0]
    cnt, cnt_offen, vol, dmax, imax, lift = _statistik(lab, F, Z0, Vf, F0, roots.size, veg_min)

    senken = []
    for p in np.nonzero((dmax >= min_tiefe) & (cnt_offen * cell / 1e4 >= min_flaeche_ha) & (vol * cell >= min_volumen))[0]:
        r, c = divmod(int(imax[p]), nc)
        sr, sc = divmod(int(roots[p]), nc)
        veg = lift[p] > max(0.5, 0.5 * dmax[p])
        senken.append(dict(typ="Vegetation" if veg else "Senke", lab=int(p), r=r, c=c, ueberlauf_r=sr, ueberlauf_c=sc,
                           volumen=float(vol[p] * cell), flaeche_ha=float(cnt[p] * cell / 1e4),
                           flaeche_offen_ha=float(cnt_offen[p] * cell / 1e4), tiefe_max=float(dmax[p]),
                           tiefe_mittel=float(vol[p] / max(cnt[p], 1)), spiegel=float(F[roots[p]]),
                           boden=float(Z0[imax[p]]), stau_baeume=float(lift[p]),
                           einzugsgebiet_ha=float(ezg[p] * cell / 1e4)))
    return senken, lab.reshape(nr, nc)
