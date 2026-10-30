#!/usr/bin/env python3
"""Öffentliche ECOSTRESS-Vorschaubilder (LST.jpeg, cloud.jpeg, ohne Anmeldung) für ausgewählte Nachtaufnahmen laden,
auf den Thurgau zuschneiden (beide MGRS-Kacheln 32TMT/32TNT zusammengesetzt) und den Wolkenanteil bestimmen.

usage: ecostress_browse.py <ecostress_nacht.csv> <out_dir> [--min-stationen 3] [--min-schnee 1]
"""
import argparse
import csv
import io
import os
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

PUB = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-public/ECO_L2T_LSTE.002/{g}/{g}{s}"
# Sentinel-2/ECOSTRESS-Kacheln: linke obere Ecke (UTM 32N), 109.8 km, 70 m -> 1568 px
UL = {"32TMT": (399960, 5300040), "32TNT": (499980, 5300040)}
RES = 70.0
# Thurgau in UTM 32N (etwas Rand)
BX0, BX1, BY0, BY1 = 485000, 550000, 5243000, 5288000
TG = ("frf", "wef", "aft", "bnu", "ste", "die", "nie", "eko", "kib")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "kaltluftseen-finder"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def crop(img, tile):
    global BX0, BX1, BY0, BY1
    """Ausschnitt BX/BY aus einer Kachel als Array im Zielraster (NaN ausserhalb)."""
    W = int((BX1 - BX0) / RES)
    H = int((BY1 - BY0) / RES)
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    n = a.shape[0]
    res = (109800.0 / n)
    ux, uy = UL[tile]
    out = np.full((H, W, 3), np.nan, np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    X = BX0 + (xs + 0.5) * RES
    Y = BY1 - (ys + 0.5) * RES
    c = ((X - ux) / res).astype(int)
    r = ((uy - Y) / res).astype(int)
    ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
    out[ok] = a[r[ok], c[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--min-stationen", type=int, default=3)
    ap.add_argument("--min-schnee", type=float, default=1)
    ap.add_argument("--box", help="Ausschnitt in UTM 32N: E0,N0,E1,N1 (Standard Thurgau)")
    ap.add_argument("--top", type=int, default=20, help="Vorschaubilder nur für die N klarsten Aufnahmen")
    a = ap.parse_args()
    global BX0, BX1, BY0, BY1
    if a.box:
        BX0, BY0, BX1, BY1 = map(float, a.box.split(","))
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv)))
    groups = {}
    for x in rows:
        s = dict(kv.split("=") for kv in x["schnee_stationen"].split() if "=" in kv)
        tg = {k: float(s[k]) for k in TG if k in s}
        if sum(v >= a.min_schnee for v in tg.values()) < a.min_stationen:
            continue
        key = x["datum"] + "T" + x["zeit_utc"][:5]
        groups.setdefault(key, {"tg": tg, "g": []})["g"].append((x["kachel"], x["granule"]))
    summ = []
    imgs = {}
    for key, d in sorted(groups.items()):
        lst = None
        cld = None
        for tile, g in d["g"]:
            if tile not in UL:
                continue
            try:
                L = crop(Image.open(io.BytesIO(fetch(PUB.format(g=g, s="_LST.jpeg")))), tile)
                C = crop(Image.open(io.BytesIO(fetch(PUB.format(g=g, s="_cloud.jpeg")))), tile)
            except Exception as e:
                print(key, g, "Fehler", e)
                continue
            lst = L if lst is None else np.where(np.isnan(lst), L, lst)
            cld = C if cld is None else np.where(np.isnan(cld), C, cld)
        if lst is None:
            continue
        valid = ~np.isnan(lst[..., 0])
        # Vorschaubild: kein Wert = schwarz; Wolkenmaske: hell = Wolke
        data = valid & (lst.max(-1) > 40)  # schwarz = keine Daten / maskiert
        cl = np.nan_to_num(cld)
        cloudy = valid & (cl[..., 0] > 150) & (cl[..., 2] < 110)  # Wolkenmaske: rot = Wolke, blau = klar
        cover = (data & ~cloudy).sum() / max(valid.sum(), 1)
        cfrac = (cloudy | ~data).sum() / max(valid.sum(), 1)
        if valid.sum() / valid.size < 0.5:
            continue
        name = key.replace(":", "")
        im = Image.fromarray(np.nan_to_num(lst).clip(0, 255).astype(np.uint8))
        im2 = Image.fromarray((np.nan_to_num(cld).clip(0, 255)).astype(np.uint8))
        both = Image.new("RGB", (im.width * 2 + 10, im.height + 30), "white")
        both.paste(im, (0, 30))
        both.paste(im2, (im.width + 10, 30))
        ImageDraw.Draw(both).text((5, 5), f"{key} UTC  Daten {cover:.0%}  Wolken {cfrac:.0%}  Schnee " +
                                  " ".join(f"{k}={v:g}" for k, v in d["tg"].items()), fill="black")
        imgs[key] = both
        summ.append((key, f"{cover:.2f}", f"{cfrac:.2f}", " ".join(f"{k}={v:g}" for k, v in d["tg"].items()),
                     " ".join(g for _, g in d["g"])))
        print(key, f"Daten {cover:.0%} Wolken {cfrac:.0%}")
    summ.sort(key=lambda r: -float(r[1]))
    for r in summ[:a.top]:
        imgs[r[0]].save(os.path.join(a.out, f"eco_{r[0].replace(':', '')}.png"))
    with open(os.path.join(a.out, "uebersicht.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["zeit_utc", "datenanteil", "wolkenanteil", "schnee_tg", "granules"])
        w.writerows(summ)


if __name__ == "__main__":
    main()
