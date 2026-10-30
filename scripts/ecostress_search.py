#!/usr/bin/env python3
"""ECOSTRESS-Nachtaufnahmen (L2T LSTE, 70 m) über einem Gebiet suchen und mit der Schneehöhe von MeteoSchweiz-Stationen
verknüpfen. Braucht keine Anmeldung (NASA CMR ist öffentlich). Ausgabe: CSV mit allen Nachtaufnahmen im Winterhalbjahr.

usage: ecostress_search.py [--point 9.1,47.6] [--out out/ecostress_nacht.csv]
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
STAC_NIME = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-nime/items"
STAC_SMN = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items"


def get(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "kaltluftseen-finder", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(), dict(r.headers)


def cmr_granules(point, start, end):
    out = []
    after = None
    while True:
        url = (f"{CMR}?short_name=ECO_L2T_LSTE&version=002&point={point}&page_size=2000"
               f"&temporal={start},{end}")
        data, h = get(url, {"CMR-Search-After": after} if after else None)
        js = json.loads(data)
        for it in js.get("items", []):
            u = it["umm"]
            name = u["GranuleUR"]
            t = u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            cloud = None
            for a in u.get("AdditionalAttributes", []) or []:
                if "cloud" in a.get("Name", "").lower():
                    cloud = (a.get("Values") or [None])[0]
            urls = [x["URL"] for x in u.get("RelatedUrls", []) if x.get("URL", "").endswith("_LST.tif")]
            out.append((name, t, cloud, urls[0] if urls else ""))
        after = h.get("CMR-Search-After") or h.get("cmr-search-after")
        if not after or not js.get("items"):
            break
    return out


def snow_stations(bbox):
    """Stationen mit Schneehöhe (hto000d0) im Gebiet: {kürzel: (name, lon, lat, url_csv)}"""
    res = {}
    for base in (STAC_NIME, STAC_SMN):
        url = f"{base}?bbox={bbox}&limit=100"
        try:
            js = json.loads(get(url)[0])
        except Exception as e:
            print("STAC", base, e, file=sys.stderr)
            continue
        for f in js.get("features", []):
            for k, a in f.get("assets", {}).items():
                if k.endswith("_d_historical.csv") or k.endswith("_d_recent.csv"):
                    res.setdefault(f["id"], [f["geometry"]["coordinates"], []])[1].append(a["href"])
    return res


def snow_series(urls):
    """Tageswerte Schneehöhe (cm) aus den MeteoSchweiz-CSV: {YYYY-MM-DD: cm}"""
    ser = {}
    for u in urls:
        try:
            txt = get(u)[0].decode("latin-1")
        except Exception as e:
            print("CSV", u, e, file=sys.stderr)
            continue
        rd = csv.DictReader(io.StringIO(txt), delimiter=";")
        for r in rd:
            v = r.get("hto000d0")
            if v in (None, "", "-"):
                continue
            d = r["reference_timestamp"][:10]
            d = f"{d[6:10]}-{d[3:5]}-{d[0:2]}" if "." in d else d
            try:
                ser[d] = float(v)
            except ValueError:
                pass
    return ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="9.1,47.6")
    ap.add_argument("--bbox", default="8.65,47.36,9.52,47.71")
    ap.add_argument("--out", default="out/ecostress_nacht.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    g = cmr_granules(a.point, "2018-07-01T00:00:00Z", time.strftime("%Y-%m-%dT00:00:00Z"))
    print(f"{len(g)} ECOSTRESS-Kacheln am Punkt {a.point}")
    sta = snow_stations(a.bbox)
    print("Stationen:", {k: v[0] for k, v in sta.items()})
    snow = {}
    for k, (xy, urls) in sta.items():
        s = snow_series(urls)
        if s:
            snow[k] = s
            print(f"  {k}: {len(s)} Tage Schneehöhe")

    rows = []
    for name, t, cloud, url in g:
        hh = int(t[11:13])
        mm = int(t[5:7])
        local = (hh + 1) % 24  # MEZ
        night = local >= 19 or local < 7
        if not night or mm not in (11, 12, 1, 2, 3):
            continue
        day = t[:10]
        if local < 7:  # Morgen: Schnee vom selben Tag; Abend: auch selber Tag
            pass
        sn = {k: s.get(day) for k, s in snow.items()}
        rows.append([day, t[11:19], f"{local:02d}:{t[14:16]}", (re.search(r"_(\d{2}[A-Z]{3})_", name) or re.search("", "")).group(0).strip("_"), cloud,
                     max([v for v in sn.values() if v is not None], default=""),
                     " ".join(f"{k}={v:g}" for k, v in sn.items() if v is not None), name, url])
    rows.sort()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datum", "zeit_utc", "zeit_mez", "kachel", "cloud", "schnee_max_cm", "schnee_stationen", "granule",
                    "url_lst"])
        w.writerows(rows)
    print(f"{len(rows)} Nachtaufnahmen Nov-Mär -> {a.out}")
    for r in rows:
        if r[5] != "" and r[5] > 0:
            print("SCHNEE", r[:7])


if __name__ == "__main__":
    main()
