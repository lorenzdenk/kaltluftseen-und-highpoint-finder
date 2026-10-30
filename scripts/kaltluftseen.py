#!/usr/bin/env python3
"""Kaltluftseen aus swissALTI3D (+ Vegetationshöhe) finden und als GPX/GeoJSON/CSV ausgeben.

Drei Arten von Kaltluftseen:
  Senke         geschlossene Mulde im Gelände: Kaltluft kann nur über den tiefsten Randpunkt abfliessen.
  Vegetation    Wald, Waldrand, Hecke staut den Abfluss. Bäume werden als teildurchlässiger Damm behandelt:
                wirksame Oberfläche = Gelände + min(k * Vegetationshöhe, max), nur wo Vegetation >= vmin.
  Talverengung  Kaltluft fliesst talab, aber eine Engstelle lässt nicht alles durch. Je Talquerschnitt
                (alle ~50 m entlang der Tiefenlinien) wird die Kaltluftmächtigkeit h gesucht, bei der der
                kritische Abfluss Q = sqrt(g' A³ / B) (A Querschnittsfläche, B Breite) gleich der im Einzugsgebiet
                produzierten Kaltluft ist (Produktionsrate * Einzugsgebiet, gedeckelt). Jeder Querschnitt wird ein
                virtueller Damm auf Höhe Tiefenlinie + h. Ein See zählt, wenn er weiter oben liegende Querschnitte
                um mindestens --min-stau Meter überstaut (Rückstau über ihr eigenes Fliessniveau).

Ausgabe in --out:
  kaltluftseen_<name>.gpx          ein Wegpunkt je See (tiefste Stelle), sortiert nach Volumen
  kaltluftseen_<name>_umrisse.gpx  Umrisse als Tracks
  kaltluftseen_<name>.geojson      Umrisse mit allen Kennzahlen (WGS84)
  kaltluftseen_<name>.csv          Tabelle

usage: kaltluftseen.py [--name thurgau] [--cache cache] [--out out] [--res 5] ...
"""
import argparse
import csv
import glob
import json
import math
import os
import time
from xml.sax.saxutils import escape

import numpy as np
from numba import njit
from osgeo import gdal, ogr, osr
from scipy import ndimage

gdal.UseExceptions()

N8R = np.array([-1, -1, -1, 0, 0, 1, 1, 1], np.int64)
N8C = np.array([-1, 0, 1, -1, 1, -1, 0, 1], np.int64)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- Priority-Flood (Barnes et al. 2014) ----

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
    """Füllt alle abflusslosen Bereiche bis zum Überlauf.
    Rückgabe: F (gefüllte Höhe), src (Index der Überlaufzelle, über die ein gefüllter/flacher Bereich abfliesst; -1
    sonst), parent (Abflussrichtung, Index der Zelle, von der aus die Zelle erreicht wurde; -1 = Rand), order
    (Bearbeitungsreihenfolge, von unten nach oben)."""
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
def accumulate(parent, order, cell_area):
    acc = np.zeros(parent.size, np.float32)
    for k in range(order.size - 1, -1, -1):
        i = order[k]
        acc[i] += cell_area
        p = parent[i]
        if p >= 0:
            acc[p] += acc[i]
    return acc


# ---------------------------------------------------------------- virtuelle Dämme an Talquerschnitten ----

@njit(cache=True)
def _section(E, P, nr, nc, r0, c0, pr, pc, step, K, lev, out_r, out_c):
    """Querschnitt bei Wasserspiegel lev über Gelände E: (wirksame Fläche in px*m, Breite in px, Anzahl Zellen in
    out_*, offen). P = Durchlässigkeit je Zelle (1 offen, kleiner in Wald/Hecke: Bäume bremsen die Kaltluft).
    offen: 0 = geschlossen, 1 = breiter als wmax oder läuft über einen Rücken ins Nachbartal über,
    2 = Rand oder keine Daten erreicht."""
    tol = max(1.0, 0.2 * (lev - E[r0 * nc + c0]))
    area = 0.0
    width = 0.0
    n = 0
    z0 = E[r0 * nc + c0]
    if z0 >= lev:
        return 0.0, 0.0, 0, 0
    # linke Seite (rückwärts), dann Mitte, dann rechte Seite: Zellen in Reihenfolge entlang der Linie
    left = 0
    mx = -1e30
    for k in range(1, K + 1):
        rr = int(round(r0 - pr * step * k))
        cc = int(round(c0 - pc * step * k))
        if rr < 0 or cc < 0 or rr >= nr or cc >= nc:
            return 0.0, 0.0, 0, 2
        e = E[rr * nc + cc]
        if not (e == e):
            return 0.0, 0.0, 0, 2
        if e >= lev:
            break
        if e > mx:
            mx = e
        elif e < mx - tol:  # Rücken überschritten: Kaltluft fliesst ins Nachbartal
            return 0.0, 0.0, 0, 1
        left = k
        area += (lev - e) * step * P[rr * nc + cc]
        width += step
        if k == K:
            return 0.0, 0.0, 0, 1
    for k in range(left + 1, 0, -1):  # inkl. erster trockener Punkt, damit der Damm am Hang anschliesst
        out_r[n] = int(round(r0 - pr * step * k))
        out_c[n] = int(round(c0 - pc * step * k))
        n += 1
    out_r[n] = r0
    out_c[n] = c0
    n += 1
    area += (lev - z0) * step * P[r0 * nc + c0]
    width += step
    mx = -1e30
    for k in range(1, K + 1):
        rr = int(round(r0 + pr * step * k))
        cc = int(round(c0 + pc * step * k))
        if rr < 0 or cc < 0 or rr >= nr or cc >= nc:
            return 0.0, 0.0, 0, 2
        e = E[rr * nc + cc]
        if not (e == e):
            return 0.0, 0.0, 0, 2
        out_r[n] = rr
        out_c[n] = cc
        n += 1
        if e >= lev:
            break
        if e > mx:
            mx = e
        elif e < mx - tol:
            return 0.0, 0.0, 0, 1
        area += (lev - e) * step * P[rr * nc + cc]
        width += step
        if k == K:
            return 0.0, 0.0, 0, 1
    return area, width, n, 0


@njit(cache=True)
def main_child(parent, acc):
    """Hauptzufluss je Zelle (Nachbar mit dem grössten Einzugsgebiet, der in die Zelle fliesst)."""
    child = np.full(parent.size, -1, np.int32)
    for i in range(parent.size):
        p = parent[i]
        if p >= 0 and (child[p] < 0 or acc[i] > acc[child[p]]):
            child[p] = i
    return child


@njit(cache=True)
def build_dams(G, P, W, dam_id, parent, child, acc, nstep, nr, nc, res, block, amin, acap, qprod, gred, hmax, wmax, maxwh):
    """Setzt virtuelle Dämme in W. Rückgabe Dammtabelle: (zelle, niveau, h, zufluss_m3s, breite_m)."""
    step = 0.5
    K = int(wmax / (res * step))
    buf_r = np.empty(2 * K + 4, np.int64)
    buf_c = np.empty(2 * K + 4, np.int64)
    nbr = (nr + block - 1) // block
    nbc = (nc + block - 1) // block
    maxd = nbr * nbc
    d_cell = np.empty(maxd, np.int64)
    d_lev = np.empty(maxd, np.float32)
    d_h = np.empty(maxd, np.float32)
    d_q = np.empty(maxd, np.float32)
    d_b = np.empty(maxd, np.float32)
    d_pr = np.empty(maxd, np.float32)
    d_pc = np.empty(maxd, np.float32)
    nd = 0
    for br in range(nbr):
        for bc in range(nbc):
            best = -1
            ba = amin
            for r in range(br * block, min((br + 1) * block, nr)):
                for c in range(bc * block, min((bc + 1) * block, nc)):
                    i = r * nc + c
                    if acc[i] >= ba and G[i] == G[i]:
                        ba = acc[i]
                        best = i
            if best < 0:
                continue
            # Fliessrichtung: von ~nstep Zellen oberhalb (Hauptzufluss) bis ~nstep Zellen unterhalb
            j = best
            for _ in range(nstep):
                if parent[j] < 0:
                    break
                j = parent[j]
            u = best
            for _ in range(nstep):
                if child[u] < 0:
                    break
                u = child[u]
            r0 = best // nc
            c0 = best - r0 * nc
            dr = float(j // nc - u // nc)
            dc = float((j - (j // nc) * nc) - (u - (u // nc) * nc))
            L = math.sqrt(dr * dr + dc * dc)
            if L < 4:
                continue
            z0 = G[best]
            Q = min(acc[best], acap) * qprod
            # Querschnitt senkrecht zur Fliessrichtung, dazu ±20° und ±40°: massgebend ist der engste (grösstes h)
            hbest = -1.0
            Qbest = Q
            pbr = 0.0
            pbc = 0.0
            for ang in (0.0, 0.35, -0.35):
                ca = math.cos(ang)
                sa = math.sin(ang)
                fr = (dr * ca - dc * sa) / L
                fc = (dr * sa + dc * ca) / L
                pr = -fc
                pc = fr
                Qa = Q
                h = 0.0
                for _it in range(4):
                    # Bisektion: kritischer Abfluss = Zufluss
                    lo = 0.0
                    hi = hmax
                    ok_hi = False
                    for _ in range(26):
                        mid = 0.5 * (lo + hi)
                        a, w, n, opn = _section(G, P, nr, nc, r0, c0, pr, pc, step, K, z0 + mid, buf_r, buf_c)
                        if opn > 0:
                            hi = mid
                            continue
                        qc = math.sqrt(gred * (a * res) ** 3 / (w * res)) if w > 0 else 0.0
                        if qc >= Qa:
                            hi = mid
                            ok_hi = True
                        else:
                            lo = mid
                    h = hi if ok_hi else lo
                    # flache Talböden: mehrere parallele Fliesslinien. Massgebend ist die grösste im Querschnitt.
                    a, w, n, opn = _section(G, P, nr, nc, r0, c0, pr, pc, step, K, z0 + h, buf_r, buf_c)
                    qs = Qa
                    for k in range(n):
                        v = min(acc[buf_r[k] * nc + buf_c[k]], acap) * qprod
                        if v > qs:
                            qs = v
                    if qs <= Qa * 1.02:
                        break
                    Qa = qs
                if h > hbest:
                    hbest = h
                    pbr = pr
                    pbc = pc
                    Qbest = Qa
            h = hbest
            if h < 0.5:
                continue
            lev = z0 + h
            a, w, n, opn = _section(G, P, nr, nc, r0, c0, pbr, pbc, step, K, lev, buf_r, buf_c)
            if opn > 0 or n == 0:
                continue
            if w * res > maxwh * h:  # kein Tal, sondern flaches/offenes Gelände: kein virtueller Damm
                continue
            nd += 1
            d_cell[nd - 1] = best
            d_lev[nd - 1] = lev
            d_h[nd - 1] = h
            d_q[nd - 1] = Qbest
            d_b[nd - 1] = w * res
            d_pr[nd - 1] = pbr
            d_pc[nd - 1] = pbc
            pr_, pc_ = buf_r[0], buf_c[0]
            for k in range(n):
                rr = buf_r[k]
                cc = buf_c[k]
                if k > 0 and rr != pr_ and cc != pc_:  # diagonaler Schritt: Lücke schliessen (4er-Nachbarschaft)
                    q = pr_ * nc + cc
                    if W[q] < lev:
                        W[q] = lev
                        dam_id[q] = nd
                q = rr * nc + cc
                if W[q] < lev:
                    W[q] = lev
                    dam_id[q] = nd
                pr_, pc_ = rr, cc
    return d_cell[:nd], d_lev[:nd], d_h[:nd], d_q[:nd], d_b[:nd], d_pr[:nd], d_pc[:nd]


# ---------------------------------------------------------------- Auswertung je See ----

@njit(cache=True)
def pool_stats(lab, F, Z, V, dam_id, d_lev, d_q, qroot, raised, npool, vmin, qfrac):
    cnt = np.zeros(npool, np.int64)
    cnt_open = np.zeros(npool, np.int64)
    vol = np.zeros(npool, np.float64)
    dmax = np.zeros(npool, np.float32)
    imax = np.full(npool, -1, np.int64)
    stau = np.zeros(npool, np.float32)
    nsub = np.zeros(npool, np.int64)
    for i in range(lab.size):
        p = lab[i]
        if p < 0 or not raised[i]:
            continue
        d = F[i] - Z[i]
        if d < 0:
            d = 0.0
        cnt[p] += 1
        if V[i] < vmin:
            cnt_open[p] += 1
        vol[p] += d
        if d > dmax[p]:
            dmax[p] = d
            imax[p] = i
        if dam_id[i] > 0 and d_q[dam_id[i] - 1] >= qfrac * qroot[p]:
            # überstauter Querschnitt weiter oben im selben Tal (ähnliche Kaltluftmenge, kein Seitenrinnsal)
            nsub[p] += 1
            ex = F[i] - d_lev[dam_id[i] - 1]
            if ex > stau[p]:
                stau[p] = ex
    return cnt, cnt_open, vol, dmax, imax, stau, nsub


@njit(cache=True)
def constriction(lab, raised, dam_id, d_cell, d_q, d_pr, d_pc, root_dam, levels, G, P, nr, nc, res, wmax, qfrac):
    """Je See: Breite der Engstelle (Abflussquerschnitt) und grösste Breite der überstauten Querschnitte weiter
    oben im selben Tal, beide auf Höhe des Seespiegels. Offener Querschnitt = wmax*2."""
    npool = root_dam.size
    nd = d_cell.size
    step = 0.5
    K = int(wmax / (res * step))
    br = np.empty(2 * K + 4, np.int64)
    bc = np.empty(2 * K + 4, np.int64)
    dam_pool = np.full(nd, -1, np.int64)
    for i in range(lab.size):
        p = lab[i]
        if p >= 0 and raised[i] and dam_id[i] > 0 and root_dam[p] > 0:
            d = dam_id[i] - 1
            if dam_pool[d] < 0 and d_q[d] >= qfrac * d_q[root_dam[p] - 1]:
                dam_pool[d] = p
    w0 = np.zeros(npool, np.float32)
    wup = np.zeros(npool, np.float32)
    for p in range(npool):
        if root_dam[p] > 0:
            d = root_dam[p] - 1
            r0 = d_cell[d] // nc
            a, w, n, opn = _section(G, P, nr, nc, r0, d_cell[d] - r0 * nc, d_pr[d], d_pc[d], step, K, levels[p], br, bc)
            w0[p] = 2 * wmax if opn > 0 else w * res
    for d in range(nd):
        p = dam_pool[d]
        if p < 0:
            continue
        r0 = d_cell[d] // nc
        a, w, n, opn = _section(G, P, nr, nc, r0, d_cell[d] - r0 * nc, d_pr[d], d_pc[d], step, K, levels[p], br, bc)
        if opn == 2:  # Rand/keine Daten: Breite unbekannt
            continue
        ww = 2 * wmax if opn == 1 else w * res
        if ww > wup[p]:
            wup[p] = ww
    return w0, wup, dam_pool


@njit(cache=True)
def stau_excess(lab, raised, dam_id, dam_pool, d_lev, F, Z, nr, nc, npool, xmin):
    """Rückstau je Zelle: Seespiegel minus Fliessniveau der Kaltluft ohne Stau. Das Fliessniveau wird von den
    überstauten Querschnitten desselben Tals (Niveau ohne Rückstau) in den See hinein übertragen (nächster Querschnitt).
    Rückgabe je See: Volumen, grösste Höhe, Zelle der grössten Höhe, Anzahl Zellen mit Rückstau >= xmin."""
    N = lab.size
    Lp = np.full(N, np.nan, np.float32)
    q = np.empty(N, np.int64)
    qt = 0
    for i in range(N):
        p = lab[i]
        if p >= 0 and raised[i] and dam_id[i] > 0 and dam_pool[dam_id[i] - 1] == p:
            Lp[i] = d_lev[dam_id[i] - 1]
            q[qt] = i
            qt += 1
    qh = 0
    while qh < qt:
        i = q[qh]
        qh += 1
        r = i // nc
        c = i - r * nc
        for k in range(8):
            rr = r + N8R[k]
            cc = c + N8C[k]
            if rr < 0 or cc < 0 or rr >= nr or cc >= nc:
                continue
            j = rr * nc + cc
            if lab[j] == lab[i] and raised[j] and not (Lp[j] == Lp[j]):
                Lp[j] = Lp[i]
                q[qt] = j
                qt += 1
    xvol = np.zeros(npool, np.float64)
    xmax = np.zeros(npool, np.float32)
    ximax = np.full(npool, -1, np.int64)
    xcnt = np.zeros(npool, np.int64)
    xmask = np.zeros(N, np.bool_)
    for i in range(N):
        p = lab[i]
        if p < 0 or not raised[i] or not (Lp[i] == Lp[i]):
            continue
        base = Z[i] if Z[i] > Lp[i] else Lp[i]
        ex = F[i] - base
        if ex <= 0:
            continue
        xvol[p] += ex
        if ex > xmax[p]:
            xmax[p] = ex
            ximax[p] = i
        if ex >= xmin:
            xcnt[p] += 1
            xmask[i] = True
    return xvol, xmax, ximax, xcnt, xmask


@njit(cache=True)
def mainstem_stau(root_dam, levels, d_cell, d_lev, dam_id, child, Z, maxsteps):
    """Rückstau entlang des Haupttals oberhalb der Engstelle (Kette der grössten Zuflüsse): grösste Differenz
    Seespiegel minus eigenes Fliessniveau der Querschnitte auf dieser Kette. Ein Seitental, das nur vom dickeren
    Kaltluftstrom des Haupttals eingestaut wird, zählt damit nicht als Engstelle."""
    npool = root_dam.size
    out = np.zeros(npool, np.float32)
    for p in range(npool):
        if root_dam[p] <= 0:
            continue
        lev = levels[p]
        j = d_cell[root_dam[p] - 1]
        for _ in range(maxsteps):
            j = child[j]
            if j < 0 or Z[j] >= lev:
                break
            d = dam_id[j]
            if d > 0 and d_cell[d - 1] == j:
                ex = lev - d_lev[d - 1]
                if ex > out[p]:
                    out[p] = ex
    return out


def label_pools(F, W, src, valid, eps):
    raised = valid & (F - W > eps) & (src >= 0)
    roots, inv = np.unique(src[raised], return_inverse=True)
    lab = np.full(F.size, -1, np.int32)
    lab[np.nonzero(raised)[0]] = inv.astype(np.int32)
    return lab, roots, raised


# ---------------------------------------------------------------- Ein-/Ausgabe ----

def srs(epsg):
    s = osr.SpatialReference()
    s.ImportFromEPSG(epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def read_grid(files, bounds, res, srcnodata, resample, dtype=gdal.GDT_Float32):
    vrt = gdal.BuildVRT("/vsimem/in.vrt", files, resolution="highest", srcNodata=srcnodata, VRTNodata=srcnodata)
    vrt = None
    ds = gdal.Warp("", "/vsimem/in.vrt", format="MEM", outputBounds=bounds, xRes=res, yRes=res,
                   resampleAlg=resample, srcNodata=srcnodata, dstNodata=np.nan, outputType=dtype, multithread=True,
                   warpOptions=["NUM_THREADS=ALL_CPUS"])
    a = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    gdal.Unlink("/vsimem/in.vrt")
    return a


def fmt_vol(v):
    if v >= 1e6:
        return f"{v / 1e6:.1f} Mio m³"
    if v >= 1e4:
        return f"{v / 1e3:.0f}'000 m³"
    return f"{v:.0f} m³"


def polygons(lab_grid, keep_ids, gt, simplify):
    """Umrisse der Seen (LV95) je ID."""
    nr, nc = lab_grid.shape
    drv = gdal.GetDriverByName("MEM")
    ds = drv.Create("", nc, nr, 1, gdal.GDT_Int32)
    ds.SetGeoTransform(gt)
    ds.SetSpatialRef(srs(2056))
    g = np.where(np.isin(lab_grid, keep_ids), lab_grid, -1).astype(np.int32)
    b = ds.GetRasterBand(1)
    b.WriteArray(g)
    m = drv.Create("", nc, nr, 1, gdal.GDT_Byte)
    m.GetRasterBand(1).WriteArray((g >= 0).astype(np.uint8))
    vds = ogr.GetDriverByName("Memory").CreateDataSource("")
    lyr = vds.CreateLayer("p", srs(2056), ogr.wkbPolygon)
    lyr.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    gdal.Polygonize(b, m.GetRasterBand(1), lyr, 0, [], callback=None)
    out = {}
    for f in lyr:
        k = f.GetField("id")
        geom = f.GetGeometryRef().Clone()
        out[k] = geom if k not in out else out[k].Union(geom)
    return {k: (v.SimplifyPreserveTopology(simplify) if simplify > 0 else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="thurgau")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="out")
    ap.add_argument("--res", type=float, default=5, help="Rechenraster in m (Standard 5)")
    ap.add_argument("--margin-km", type=float, default=2)
    # Vegetation
    ap.add_argument("--veg-k", type=float, default=0.4, help="wirksamer Anteil der Vegetationshöhe als Damm")
    ap.add_argument("--veg-min", type=float, default=3, help="Vegetation ab dieser Höhe (m) staut")
    ap.add_argument("--veg-max", type=float, default=12, help="höchste wirksame Dammhöhe durch Vegetation (m)")
    ap.add_argument("--veg-por", type=float, default=0.3,
                    help="Talverengung: Durchlässigkeit von Wald/Hecke im Querschnitt (1 = wie offenes Gelände)")
    # Talverengung
    ap.add_argument("--dam-spacing", type=float, default=20, help="Abstand der Querschnitte (m)")
    ap.add_argument("--amin", type=float, default=0.1, help="Querschnitte ab Einzugsgebiet (km²)")
    ap.add_argument("--acap", type=float, default=10, help="Einzugsgebiet höchstens (km²) für die Kaltluftmenge")
    ap.add_argument("--prod", type=float, default=3,
                    help="wirksame Kaltluftproduktion (m³ je m² und Stunde); 3 passt zum Kaltluftsee bei Geboltshausen")
    ap.add_argument("--dT", type=float, default=3, help="Temperaturdefizit der Kaltluft (K)")
    ap.add_argument("--hmax", type=float, default=40, help="grösste Kaltluftmächtigkeit (m)")
    ap.add_argument("--wmax", type=float, default=1000, help="halbe Querschnittsbreite höchstens (m)")
    ap.add_argument("--max-wh", type=float, default=150,
                    help="Querschnitt nur, wenn Breite höchstens so viel mal die Kaltluftmächtigkeit (sonst offenes Gelände)")
    ap.add_argument("--qfrac", type=float, default=0.3,
                    help="Rückstau nur gegen Querschnitte mit mind. diesem Anteil der Kaltluftmenge am Abfluss")
    ap.add_argument("--min-widen", type=float, default=2.0,
                    help="Talverengung: Tal oberhalb auf Seespiegelhöhe mind. so viel mal breiter als die Engstelle")
    ap.add_argument("--stem-km", type=float, default=3, help="Haupttal so weit oberhalb der Engstelle prüfen (km)")
    ap.add_argument("--min-stau", type=float, default=2.0, help="Talverengung: Mindest-Rückstau (m)")
    # Filter
    ap.add_argument("--min-depth", type=float, default=1.0, help="Mindesttiefe (m)")
    ap.add_argument("--min-area", type=float, default=0.2, help="Mindestfläche offenes Gelände (ha)")
    ap.add_argument("--min-volume", type=float, default=1000, help="Mindestvolumen (m³)")
    ap.add_argument("--no-dams", action="store_true", help="Talverengungen nicht rechnen")
    ap.add_argument("--all", action="store_true", help="nicht auf Kantonsgrenze beschränken")
    ap.add_argument("--top", type=int, default=200, help="zusätzliche GPX-Datei mit den N grössten Seen")
    ap.add_argument("--window", help="nur Ausschnitt rechnen: x0,y0,x1,y1 in LV95 (m)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    bpath = os.path.join(a.cache, f"boundary_{a.name}.geojson")
    border = None
    dtm_files = sorted(glob.glob(os.path.join(a.cache, "dtm2_*.tif")))
    if not dtm_files:
        raise SystemExit("keine Höhenmodell-Kacheln im Cache: zuerst fetch.py laufen lassen")
    if os.path.exists(bpath):
        ds = ogr.Open(bpath)
        for f in ds.GetLayer(0):
            g = f.GetGeometryRef().Clone()
            border = g if border is None else border.Union(g)
        x0, x1, y0, y1 = border.GetEnvelope()
    else:
        ext = [gdal.Info(f, format="json")["cornerCoordinates"] for f in dtm_files]
        x0 = min(e["lowerLeft"][0] for e in ext); y0 = min(e["lowerLeft"][1] for e in ext)
        x1 = max(e["upperRight"][0] for e in ext); y1 = max(e["upperRight"][1] for e in ext)
    if a.window:
        x0, y0, x1, y1 = map(float, a.window.split(","))
    m = a.margin_km * 1000
    r = a.res
    bounds = [math.floor((x0 - m) / r) * r, math.floor((y0 - m) / r) * r,
              math.ceil((x1 + m) / r) * r, math.ceil((y1 + m) / r) * r]
    gt = (bounds[0], r, 0, bounds[3], 0, -r)

    log(f"Höhenmodell lesen ({len(dtm_files)} Kacheln), Raster {r:g} m")
    Z = read_grid(dtm_files, bounds, r, -9999, "average")
    nr, nc = Z.shape
    log(f"  {nc} x {nr} Zellen")
    veg_files = sorted(glob.glob(os.path.join(a.cache, "veg_*.tif")))
    if veg_files:
        log(f"Vegetationshöhe lesen ({len(veg_files)} Kacheln)")
        V = read_grid(veg_files, bounds, r, 255, "average") / 2.0
        V[~np.isfinite(V)] = 0
    else:
        log("keine Vegetationshöhe im Cache: nur Gelände")
        V = np.zeros_like(Z)
    Z = Z.ravel()
    V = V.ravel()
    valid = np.isfinite(Z)
    barrier = np.where(V >= a.veg_min, np.minimum(V * a.veg_k, a.veg_max), 0).astype(np.float32)
    E = np.where(valid, Z + barrier, np.nan).astype(np.float32)

    results = []  # (art, lab, F, roots, stats...)
    cell = r * r

    # 1) Senken und Vegetationsstau
    log("Füllen: Gelände + Vegetation")
    F1, src1, parent, order = priority_flood(E, valid, nr, nc)
    del parent, order
    lab1, roots1, raised1 = label_pools(F1, E, src1, valid, 0.05)
    dam_id = np.zeros(Z.size, np.int32)
    z1 = np.zeros(1, np.float32)
    st1 = pool_stats(lab1, F1, Z, V, dam_id, z1, z1, np.zeros(roots1.size, np.float32), raised1, roots1.size,
                     a.veg_min, 0.0)
    # Vegetation, wenn der Seespiegel ohne Bäume tiefer läge (nicht nur nach der Überlaufstelle beurteilt)
    F0, _, _, _ = priority_flood(np.where(valid, Z, np.nan).astype(np.float32), valid, nr, nc)
    lift = np.zeros(roots1.size, np.float32)
    rmask = lab1 >= 0
    np.maximum.at(lift, lab1[rmask], (F1 - F0)[rmask])
    del F0, rmask
    kinds1 = np.where(lift > np.maximum(0.5, 0.5 * st1[3]), "Vegetation", "Senke")
    log(f"  {roots1.size} gefüllte Bereiche")

    # 2) Talverengungen
    damtab = None
    if not a.no_dams:
        log("Fliesswege und Einzugsgebiete")
        _, _, parentZ, orderZ = priority_flood(np.where(valid, Z, np.nan).astype(np.float32), valid, nr, nc)
        acc = accumulate(parentZ, orderZ, np.float32(cell))
        del orderZ
        log("Talquerschnitte und virtuelle Dämme")
        W = E.copy()
        gred = 9.81 * a.dT / 283.0
        child = main_child(parentZ, acc)
        P = np.where(V >= a.veg_min, np.float32(a.veg_por), np.float32(1)).astype(np.float32)
        damtab = build_dams(Z, P, W, dam_id, parentZ, child, acc, max(int(round(60 / r)), 4), nr, nc, r, max(int(round(a.dam_spacing / r)), 1),
                            a.amin * 1e6, a.acap * 1e6, a.prod / 3600.0, gred, a.hmax, a.wmax, a.max_wh)
        del parentZ
        log(f"  {damtab[0].size} Querschnitte, Mächtigkeit Median {np.median(damtab[2]) if damtab[2].size else 0:.1f} m")
        log("Füllen: mit Dämmen")
        F2, src2, _, _ = priority_flood(W, valid, nr, nc)
        lab2, roots2, raised2 = label_pools(F2, W, src2, valid, 0.05)
        rd = dam_id[roots2]
        qroot = np.where(rd > 0, damtab[3][np.maximum(rd - 1, 0)], np.float32(0)).astype(np.float32)
        st2 = pool_stats(lab2, F2, Z, V, dam_id, damtab[1], damtab[3], qroot, raised2, roots2.size, a.veg_min,
                         a.qfrac)
        w0, wup, dam_pool = constriction(lab2, raised2, dam_id, damtab[0], damtab[3], damtab[5], damtab[6],
                                         rd.astype(np.int64), F2[roots2], Z, P, nr, nc, r, a.wmax, a.qfrac)
        xst = list(stau_excess(lab2, raised2, dam_id, dam_pool, damtab[1], F2, Z, nr, nc, roots2.size, 0.5))
        # schmale Streifen (z.B. entlang virtueller Dämme) zählen nicht: Öffnung mit 3x3
        xm = ndimage.binary_opening(xst[4].reshape(nr, nc), structure=np.ones((3, 3), bool)).ravel()
        xst[4] = xm
        xst[3] = np.bincount(lab2[xm], minlength=roots2.size)
        del xm
        mstau = mainstem_stau(rd, F2[roots2], damtab[0], damtab[1], dam_id, child, Z, int(a.stem_km * 1000 / r))
        del child
        log(f"  {roots2.size} gefüllte Bereiche")
        del W, src2

    # Auswahl
    to84 = osr.CoordinateTransformation(srs(2056), srs(4326))

    def xy(i):
        rr, cc = divmod(int(i), nc)
        return bounds[0] + (cc + 0.5) * r, bounds[3] - (rr + 0.5) * r

    pools = []

    def collect(st, roots, F, kinds, lab, which, w0=None, wup=None, xst=None, mstau=None):
        cnt, cnt_open, vol, dmax, imax, stau, nsub = st
        for p in range(roots.size):
            kind = kinds[p]
            if kind == "x":
                continue
            v_total = vol[p] * cell
            dmax_total = float(dmax[p])
            if kind == "Talverengung":
                # nur der Rückstau zählt, nicht die ohnehin fliessende Kaltluftschicht
                if nsub[p] == 0 or wup[p] < a.min_widen * w0[p] or xst[1][p] < a.min_stau or mstau[p] < a.min_stau:
                    continue
                v, dm, im, n_area, n_open = xst[0][p] * cell, float(xst[1][p]), xst[2][p], xst[3][p], xst[3][p]
            else:
                v, dm, im, n_area, n_open = v_total, dmax_total, imax[p], cnt[p], cnt_open[p]
            if dm < a.min_depth or n_open * cell / 1e4 < a.min_area or v < a.min_volume:
                continue
            X, Y = xy(im)
            if border is not None and not a.all:
                pt = ogr.Geometry(ogr.wkbPoint); pt.AddPoint_2D(X, Y)
                if not border.Contains(pt):
                    continue
            sx, sy = xy(roots[p])
            pools.append(dict(kind=kind, which=which, lab=p, X=X, Y=Y, area=n_area * cell / 1e4,
                              area_open=n_open * cell / 1e4, vol=v, dmax=dm, vol_total=v_total, dmax_total=dmax_total,
                              dmean=v / max(n_area * cell, 1), level=float(F[roots[p]]), ground=float(Z[im]),
                              stau=float(stau[p]), w0=float(w0[p]) if w0 is not None else 0.0,
                              wup=float(wup[p]) if wup is not None else 0.0,
                              lift=float(lift[p]) if which == 1 else 0.0, sx=sx, sy=sy, veg_spill=float(V[roots[p]])))

    collect(st1, roots1, F1, kinds1, lab1, 1)
    if damtab is not None:
        kinds2 = np.where(dam_id[roots2] > 0, "Talverengung", "x")
        collect(st2, roots2, F2, kinds2, lab2, 2, w0, wup, xst, mstau)
    pools.sort(key=lambda d: -d["vol"])
    log(f"{len(pools)} Kaltluftseen: " + ", ".join(f"{k} {sum(p['kind'] == k for p in pools)}"
                                                 for k in ("Senke", "Vegetation", "Talverengung")))

    # Umrisse
    simp = r * 0.75
    outl = {}
    # Talverengung: Umriss nur, wo Rückstau >= 0,5 m (ohne die fliessende Schicht)
    for which, lab in ((1, lab1), (2, np.where(xst[4], lab2, -1) if damtab is not None else None)):
        if lab is None:
            continue
        ids = [p["lab"] for p in pools if p["which"] == which]
        if ids:
            outl[which] = polygons(lab.reshape(nr, nc), np.array(ids), gt, simp)

    for k, p in enumerate(pools, 1):
        p["id"] = k
        lon, lat, _ = to84.TransformPoint(p["X"], p["Y"])
        slon, slat, _ = to84.TransformPoint(p["sx"], p["sy"])
        p.update(lon=lon, lat=lat, slon=slon, slat=slat)
        g = outl.get(p["which"], {}).get(p["lab"])
        if g is not None:
            g = g.Clone(); g.Transform(to84)
        p["geom"] = g
        reason = {"Senke": "geschlossene Mulde",
                  "Vegetation": f"Abfluss durch Wald/Hecke gestaut, Bäume heben den Seespiegel um {p['lift']:.1f} m",
                  "Talverengung": f"Engstelle ({p['w0']:.0f} m breit, Tal oberhalb "
                                  f"{'über ' + str(int(2 * a.wmax)) if p['wup'] >= 2 * a.wmax else round(p['wup'])} m), "
                                  f"Rückstau über das Fliessniveau max. {p['dmax']:.1f} m; mit fliessender Kaltluft "
                                  f"{fmt_vol(p['vol_total'])}, bis {p['dmax_total']:.0f} m tief"}[p["kind"]]
        p["name"] = f"KLS {k} {fmt_vol(p['vol'])}"
        p["desc"] = (f"{p['kind']}: {reason}. Volumen {fmt_vol(p['vol'])}, Fläche {p['area']:.1f} ha, "
                     f"Tiefe max. {p['dmax']:.1f} m / Ø {p['dmean']:.1f} m, Seespiegel {p['level']:.0f} m ü. M., "
                     f"Abfluss bei {slat:.5f}, {slon:.5f}")

    write_outputs(a, pools)
    log(f"fertig in {(time.time() - t0) / 60:.1f} min")


def write_outputs(a, pools):
    base = os.path.join(a.out, f"kaltluftseen_{a.name}")
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="kaltluftseen-finder" '
            'xmlns="http://www.topografix.com/GPX/1/1">\n<metadata><name>Kaltluftseen %s</name>'
            '<desc>Datenquelle: swissALTI3D © swisstopo; Vegetationshöhenmodell LiDAR LFI, BAFU/WSL. Modellrechnung, '
            'nicht gemessen.</desc></metadata>\n' % escape(a.name.capitalize()))
    for path, sel in ((base + ".gpx", pools), (base + f"_top{a.top}.gpx", pools[:a.top])):
        with open(path, "w") as f:
            f.write(head)
            for p in sel:
                f.write(f'<wpt lat="{p["lat"]:.6f}" lon="{p["lon"]:.6f}"><ele>{p["ground"]:.1f}</ele>'
                        f'<name>{escape(p["name"])}</name><desc>{escape(p["desc"])}</desc>'
                        f'<type>{p["kind"]}</type></wpt>\n')
            f.write("</gpx>\n")
    with open(base + "_umrisse.gpx", "w") as f:
        f.write(head)
        for p in pools:
            g = p["geom"]
            if g is None:
                continue
            f.write(f'<trk><name>{escape(p["name"])}</name><desc>{escape(p["desc"])}</desc>'
                    f'<type>{p["kind"]}</type>')
            polys = [g.GetGeometryRef(i) for i in range(g.GetGeometryCount())] \
                if g.GetGeometryType() in (ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D) else [g]
            for poly in polys:
                ring = poly.GetGeometryRef(0)  # nur Aussenrand
                f.write("<trkseg>" + "".join(f'<trkpt lat="{ring.GetY(i):.6f}" lon="{ring.GetX(i):.6f}"/>'
                                             for i in range(ring.GetPointCount())) + "</trkseg>")
            f.write("</trk>\n")
        f.write("</gpx>\n")
    keys = ["id", "kind", "vol", "area", "area_open", "dmax", "dmean", "vol_total", "dmax_total", "w0", "wup", "level", "ground", "lat", "lon",
            "slat", "slon", "veg_spill", "lift"]
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "typ", "volumen_m3", "flaeche_ha", "flaeche_offen_ha", "tiefe_max_m", "tiefe_mittel_m",
                    "volumen_inkl_fliessend_m3", "tiefe_inkl_fliessend_m", "engstelle_breite_m", "tal_oberhalb_breite_m", "seespiegel_m", "boden_tiefster_punkt_m", "lat", "lon", "abfluss_lat",
                    "abfluss_lon", "vegetation_abfluss_m", "stau_durch_baeume_m"])
        for p in pools:
            w.writerow([p["id"], p["kind"], round(p["vol"]), round(p["area"], 2), round(p["area_open"], 2),
                        round(p["dmax"], 2), round(p["dmean"], 2), *((round(p["vol_total"]), round(p["dmax_total"], 1), round(p["w0"]), round(p["wup"]))
                          if p["kind"] == "Talverengung" else ("", "", "", "")),
                        round(p["level"], 1), round(p["ground"], 1), f"{p['lat']:.6f}", f"{p['lon']:.6f}",
                        f"{p['slat']:.6f}", f"{p['slon']:.6f}", round(p["veg_spill"], 1), round(p["lift"], 1)])
    feats = []
    for p in pools:
        if p["geom"] is None:
            continue
        props = {k: (round(p[k], 6 if k in ("lat", "lon", "slat", "slon") else 2) if isinstance(p[k], float) else p[k]) for k in keys}
        props.update(name=p["name"], beschreibung=p["desc"])
        feats.append({"type": "Feature", "properties": props, "geometry": json.loads(p["geom"].ExportToJson())})
    with open(base + ".geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
    log("geschrieben:", ", ".join(os.path.basename(base) + s for s in (".gpx", f"_top{a.top}.gpx", "_umrisse.gpx", ".csv", ".geojson")))


if __name__ == "__main__":
    main()
