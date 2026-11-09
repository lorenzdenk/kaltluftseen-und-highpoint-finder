"""Kaltluftabfluss-Modell: eine Kaltluftschicht, die auf dem Gelände entsteht, hangab fliesst und sich staut.

Gerechnet wird eine tiefengemittelte Schicht (Flachwassergleichungen mit reduzierter Schwere g' = g * dT / T, ohne
Advektionsterm: "local inertial", Bates et al. 2010 / de Almeida et al. 2012), wie in den gängigen Kaltluftabfluss-
Modellen (z.B. KLAM_21 des DWD) als Ein-Schicht-Modell:

    dh/dt + div(q) = P                      h Schichtdicke (m), q Volumenstrom je m Breite (m²/s), P Produktion (m/s)
    dq/dt = -g' h grad(z + h) - cd |q| q / h²

P hängt von der Landnutzung ab (Freiland viel, Wald mittel, Siedlung wenig, Wasser nichts), der Widerstand cd von
Bodenrauigkeit und Hindernissen (Wald, Gebäude): cd = cd_grund + cd_hindernis * Anteil * min(1, Hindernishöhe / h).
Am Rand des Gebiets und an Datenlücken fliesst die Kaltluft frei ab.

Vereinfachungen: konstantes Temperaturdefizit (kein Wärmehaushalt), kein übergeordneter Wind, keine Einmischung von
oben. Das Modell beschreibt windschwache Strahlungsnächte.
"""
import math

import numpy as np
from numba import njit, prange


@njit(inline="always")
def _fluss(zl, hl, zr, hr, q, qa, qb, cdl, cdr, col, cor, obl, obr, dt, dx, gr, hmin, theta, frmax):
    """Neuer Volumenstrom über die Fläche zwischen linker und rechter Zelle (positiv = nach rechts)."""
    el = zl + hl
    er = zr + hr
    hf = max(el, er) - max(zl, zr)
    if hf <= hmin:
        return 0.0
    cd = 0.5 * (cdl + cdr)
    co = 0.5 * (col + cor)
    if co > 0.0:
        ob = 0.5 * (obl + obr)
        cd += co * (ob / hf if hf > ob else 1.0)
    qs = theta * q + 0.5 * (1.0 - theta) * (qa + qb)
    qn = (qs - gr * hf * dt * (er - el) / dx) / (1.0 + dt * cd * abs(q) / (hf * hf))
    qmax = frmax * hf * math.sqrt(gr * hf)
    if qn > qmax:
        qn = qmax
    elif qn < -qmax:
        qn = -qmax
    return qn


@njit(parallel=True, cache=True)
def _schritt(z, h, qxa, qya, qx, qy, s, cd0, cdo, hob, prod, valid, dt, dx, gr, hmin, theta, frmax, hcap, ax, ay,
             mitteln):
    """Ein Zeitschritt: alte Flüsse qxa/qya -> neue qx/qy, dann h. qx[r, c]: Fluss von Zelle (r, c-1) nach (r, c);
    qy[r, c]: von (r-1, c) nach (r, c). Rückgabe: grösste Schichtdicke."""
    nr, nc = z.shape
    for r in prange(nr):
        for c in range(nc + 1):
            a = c > 0 and valid[r, c - 1]
            b = c < nc and valid[r, c]
            if a and b:
                q = qxa[r, c]
                qa = qxa[r, c - 1] if c > 1 else q
                qb = qxa[r, c + 1] if c < nc - 1 else q
                qx[r, c] = _fluss(z[r, c - 1], h[r, c - 1], z[r, c], h[r, c], q, qa, qb, cd0[r, c - 1],
                                  cd0[r, c], cdo[r, c - 1], cdo[r, c], hob[r, c - 1], hob[r, c], dt, dx, gr, hmin,
                                  theta, frmax)
            elif a:  # freier Abfluss nach aussen
                hh = h[r, c - 1]
                qx[r, c] = hh * math.sqrt(gr * hh) if hh > hmin else 0.0
            elif b:
                hh = h[r, c]
                qx[r, c] = -hh * math.sqrt(gr * hh) if hh > hmin else 0.0
            else:
                qx[r, c] = 0.0
    for r in prange(nr + 1):
        for c in range(nc):
            a = r > 0 and valid[r - 1, c]
            b = r < nr and valid[r, c]
            if a and b:
                q = qya[r, c]
                qa = qya[r - 1, c] if r > 1 else q
                qb = qya[r + 1, c] if r < nr - 1 else q
                qy[r, c] = _fluss(z[r - 1, c], h[r - 1, c], z[r, c], h[r, c], q, qa, qb, cd0[r - 1, c],
                                  cd0[r, c], cdo[r - 1, c], cdo[r, c], hob[r - 1, c], hob[r, c], dt, dx, gr, hmin,
                                  theta, frmax)
            elif a:
                hh = h[r - 1, c]
                qy[r, c] = hh * math.sqrt(gr * hh) if hh > hmin else 0.0
            elif b:
                hh = h[r, c]
                qy[r, c] = -hh * math.sqrt(gr * hh) if hh > hmin else 0.0
            else:
                qy[r, c] = 0.0
    # Begrenzer: aus keiner Zelle fliesst mehr ab, als drin ist
    for r in prange(nr):
        for c in range(nc):
            if not valid[r, c]:
                s[r, c] = 1.0
                continue
            ab = max(qx[r, c + 1], 0.0) - min(qx[r, c], 0.0) + max(qy[r + 1, c], 0.0) - min(qy[r, c], 0.0)
            da = h[r, c] * dx / dt
            s[r, c] = 1.0 if ab <= da else da / ab
    zeilen_max = np.zeros(nr, np.float32)
    for r in prange(nr):
        hm = 0.0
        for c in range(nc):
            if not valid[r, c]:
                continue
            sc = s[r, c]
            fl = qx[r, c]
            fl *= sc if fl < 0 else (s[r, c - 1] if c > 0 and valid[r, c - 1] else 1.0)
            fr = qx[r, c + 1]
            fr *= sc if fr > 0 else (s[r, c + 1] if c < nc - 1 and valid[r, c + 1] else 1.0)
            ft = qy[r, c]
            ft *= sc if ft < 0 else (s[r - 1, c] if r > 0 and valid[r - 1, c] else 1.0)
            fb = qy[r + 1, c]
            fb *= sc if fb > 0 else (s[r + 1, c] if r < nr - 1 and valid[r + 1, c] else 1.0)
            hn = h[r, c] + dt * (prod[r, c] - (fr - fl + fb - ft) / dx)
            if hn < 0.0:
                hn = 0.0
            elif hn > hcap:
                hn = hcap
            h[r, c] = hn
            if hn > hm:
                hm = hn
            if mitteln:
                ax[r, c] += dt * 0.5 * (fl + fr)
                ay[r, c] += dt * 0.5 * (ft + fb)
        zeilen_max[r] = hm
    return zeilen_max.max()


@njit(cache=True)
def _laufe(z, h, qx, qy, qx2, qy2, s, cd0, cdo, hob, prod, valid, dauer, dx, gr, hmin, theta, frmax, hcap, alpha,
           ax, ay, mitteln, hmax):
    """Rechnet 'dauer' Sekunden weiter (qx2/qy2 sind Arbeitsfelder). Rückgabe: (Anzahl Schritte, grösste Dicke)."""
    t = 0.0
    n = 0
    alt_x, alt_y, neu_x, neu_y = qx, qy, qx2, qy2
    while t < dauer - 1e-6:
        dt = alpha * dx / math.sqrt(gr * max(hmax, 1.0))
        if dt > dauer - t:
            dt = dauer - t
        hmax = _schritt(z, h, alt_x, alt_y, neu_x, neu_y, s, cd0, cdo, hob, prod, valid, dt, dx, gr, hmin, theta,
                        frmax, hcap, ax, ay, mitteln)
        alt_x, alt_y, neu_x, neu_y = neu_x, neu_y, alt_x, alt_y
        t += dt
        n += 1
    if n % 2 == 1:  # aktueller Stand liegt in den Arbeitsfeldern
        qx[:] = qx2
        qy[:] = qy2
    return n, hmax


class Parameter:
    """Modellparameter (Standardwerte aus der Literatur zu Kaltluftabflüssen, VDI 3787 Blatt 5)."""
    dT = 3.0              # Temperaturdefizit der Kaltluftschicht gegenüber der Umgebung (K)
    T = 278.0             # Lufttemperatur (K)
    prod_frei = 12.0      # Kaltluftproduktion Freiland: Wiese, Acker (m³ je m² und Stunde)
    prod_wald = 8.0       # Wald
    prod_siedlung = 1.0   # Siedlung
    prod_wasser = 0.0     # Seen, Flüsse
    cd_grund = 0.012      # Widerstandsbeiwert Freiland inkl. Reibung an der Schichtobergrenze
    cd_wald = 0.12        # zusätzlicher Widerstand im Bestand (wirkt bis Baumhöhe)
    cd_siedlung = 0.10    # zusätzlicher Widerstand in Siedlungen
    h_siedlung = 8.0      # mittlere Gebäudehöhe (m)
    h_max = 200.0         # Obergrenze der Schichtdicke (m)
    alpha = 0.45          # CFL-Zahl
    theta = 0.9           # Glättung des Flusses (de Almeida et al. 2012); 1 = keine
    fr_max = 2.0          # höchste Froude-Zahl
    h_min = 0.01          # kleinste fliessende Schichtdicke (m)

    def __init__(self, **kw):
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"unbekannter Parameter {k}")
            setattr(self, k, v)

    @property
    def g_red(self):
        return 9.81 * self.dT / self.T


def felder(par, valid, wald, wald_h, siedlung=None, wasser=None):
    """Produktion (m/s) und Widerstandsfelder aus den Flächenanteilen je Zelle (0..1)."""
    siedlung = np.zeros_like(wald) if siedlung is None else siedlung
    wasser = np.zeros_like(wald) if wasser is None else wasser
    wald = np.clip(wald, 0, 1) * (1 - wasser)
    siedlung = np.clip(siedlung, 0, 1 - wald - wasser)
    frei = np.clip(1 - wald - siedlung - wasser, 0, 1)
    prod = (frei * par.prod_frei + wald * par.prod_wald + siedlung * par.prod_siedlung
            + wasser * par.prod_wasser) / 3600.0
    prod = np.where(valid, prod, 0).astype(np.float32)
    cdo = (wald * par.cd_wald + siedlung * par.cd_siedlung).astype(np.float32)
    anteil = np.maximum(wald + siedlung, 1e-6)
    hob = ((wald * wald_h + siedlung * par.h_siedlung) / anteil).astype(np.float32)
    cd0 = np.full(wald.shape, par.cd_grund, np.float32)
    return prod, cd0, cdo, hob


def simuliere(z, dx, stunden, par=None, prod=None, cd0=None, cdo=None, hob=None, beginn_h=5.0,
              mittel_min=10.0, schritt_min=10.0, log=None):
    """Simuliert die Kaltluftschicht ab Sonnenuntergang (h = 0).

    z: Gelände (2D, NaN = keine Daten), dx: Zellgrösse (m), stunden: Zeitpunkte der Auswertung (aufsteigend).
    Rückgabe: dict mit 'zustand' {stunde: dict(h, qx, qy, tempo)}, 'beginn' (Stunden, bis die Schicht beginn_h Meter
    dick ist; NaN = nie), 'prod', 'bilanz'.
      qx, qy = mittlerer Volumenstrom je m Breite in den letzten mittel_min Minuten (m²/s; x nach Osten, y nach Süden).
      tempo  = mittlere Fliessgeschwindigkeit der Schicht (m/s).
    """
    par = par or Parameter()
    valid = np.isfinite(z)
    zz = np.where(valid, z, 0).astype(np.float32)
    nr, nc = z.shape
    if prod is None:
        prod = np.where(valid, par.prod_frei / 3600.0, 0).astype(np.float32)
    cd0 = np.full(z.shape, par.cd_grund, np.float32) if cd0 is None else cd0
    cdo = np.zeros(z.shape, np.float32) if cdo is None else cdo
    hob = np.zeros(z.shape, np.float32) if hob is None else hob
    h = np.zeros(z.shape, np.float32)
    qx = np.zeros((nr, nc + 1), np.float32)
    qy = np.zeros((nr + 1, nc), np.float32)
    qx2 = np.zeros_like(qx)
    qy2 = np.zeros_like(qy)
    s = np.ones(z.shape, np.float32)
    ax = np.zeros(z.shape, np.float32)
    ay = np.zeros(z.shape, np.float32)
    beginn = np.full(z.shape, np.nan, np.float32)
    gr = np.float32(par.g_red)
    zeiten = sorted(float(x) * 3600.0 for x in stunden)
    mittel = mittel_min * 60.0
    out = {}
    t = 0.0
    hmax = 0.0
    nges = 0
    for tz in zeiten:
        # Abschnitte: normal bis tz - mittel, dann Mittelungsfenster
        ax[:] = 0
        ay[:] = 0
        t_fenster = 0.0
        while t < tz - 1e-6:
            im_fenster = t >= tz - mittel - 1e-6
            ende = tz if im_fenster else min(t + schritt_min * 60.0, tz - mittel)
            n, hmax = _laufe(zz, h, qx, qy, qx2, qy2, s, cd0, cdo, hob, prod, valid, ende - t, dx, gr, par.h_min, par.theta,
                             par.fr_max, par.h_max, par.alpha, ax, ay, im_fenster, hmax)
            if im_fenster:
                t_fenster += ende - t
            t = ende
            nges += n
            neu = valid & np.isnan(beginn) & (h >= beginn_h)
            beginn[neu] = t / 3600.0
            if log:
                log(f"  {t / 3600:5.2f} h: {nges} Schritte, Schicht max. {hmax:.0f} m, "
                    f"Mittel {float(h[valid].mean()):.1f} m")
        fx = ax / max(t_fenster, 1e-6)
        fy = ay / max(t_fenster, 1e-6)
        hh = np.where(valid, h, np.nan).astype(np.float32)
        strom = np.hypot(fx, fy)
        tempo = np.where(hh > 0.5, strom / np.maximum(hh, 0.5), 0).astype(np.float32)
        out[tz / 3600.0] = dict(h=hh,
                                qx=np.where(valid, fx, np.nan).astype(np.float32),
                                qy=np.where(valid, fy, np.nan).astype(np.float32),
                                tempo=np.where(valid, tempo, np.nan).astype(np.float32))
    erzeugt = float(prod.sum(dtype=np.float64)) * t * dx * dx
    vorhanden = float(h[valid].sum(dtype=np.float64)) * dx * dx
    return dict(zustand=out, beginn=beginn, prod=prod, valid=valid,
                bilanz=dict(erzeugt_m3=erzeugt, vorhanden_m3=vorhanden, schritte=nges))
