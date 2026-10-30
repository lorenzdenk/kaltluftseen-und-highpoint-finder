#!/usr/bin/env python3
"""Eingangsdaten für ein Gebiet laden und im Cache-Verzeichnis ablegen. Was schon da ist, wird nicht nochmals geladen.

Cache-Inhalt (flach, damit er 1:1 als GitHub-Release abgelegt werden kann):
  boundary_<name>.geojson   Kantonsgrenze (swissBOUNDARIES3D via api3.geo.admin.ch), LV95
  alti_index_<name>.json    Liste der swissALTI3D-Kacheln (STAC), je 1-km-Kachel neuester Jahrgang
  dtm2_<E>_<N>.tif          swissALTI3D 2 m, 10 x 10 km (E, N = linke untere Ecke in km), Float32, NoData -9999
  veg_<...>.tif             Vegetationshöhe (Byte, Wert = Höhe * 2, 255 = keine Daten), 10 x 10 km

Vegetationshöhe: zuerst aus dem Release "ch-vhm-<name>" des Repos render-baumhoehen-overlay (1 m, schon aufbereitet),
sonst direkt aus dem Vegetationshöhenmodell LiDAR LFI (BAFU/WSL) per HTTP-Range, je Pixel neuester Jahrgang, 2 m.

usage: fetch.py [--name thurgau] [--canton 20] [--cache cache] [--margin-km 2] [--threads 8]
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
BOUNDARY = ("https://api3.geo.admin.ch/rest/services/api/MapServer/ch.swisstopo.swissboundaries3d-kanton-flaeche.fill/"
            "{canton}?geometryFormat=geojson&sr=2056&returnGeometry=true")
VHM_RELEASE = "https://github.com/lorenzdenk/render-baumhoehen-overlay/releases/download/ch-vhm-{name}/"
VHM_COG = ("/vsicurl/https://data.geo.admin.ch/ch.bafu.landesforstinventar-vegetationshoehenmodell_lidar/"
           "landesforstinventar-vegetationshoehenmodell_lidar_{y}/landesforstinventar-vegetationshoehenmodell_lidar_{y}_2056.tif")
VHM_YEARS = range(2026, 2016, -1)
TILE = 10000  # Cache-Kacheln 10 km
UA = {"User-Agent": "kaltluftseen-finder/1.0"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http_get(url, path=None, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
            if path is None:
                return data
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
            return path
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except Exception as e:  # Netzfehler: nochmals
            err = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url}: {err}")


def boundary(a):
    """Kantonsgrenze laden (gecacht). Gibt (ogr-Geometrie LV95 oder None, bbox LV95) zurück."""
    path = os.path.join(a.cache, f"boundary_{a.name}.geojson")
    if not os.path.exists(path) and a.canton:
        try:
            js = json.loads(http_get(BOUNDARY.format(canton=a.canton)))
            feat = js["feature"]
            fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:2056"}},
                  "features": [{"type": "Feature", "properties": feat.get("properties", {}), "geometry": feat["geometry"]}]}
            with open(path, "w") as f:
                json.dump(fc, f)
            log("Kantonsgrenze geladen:", feat.get("properties", {}).get("name", a.canton))
        except Exception as e:
            log("Kantonsgrenze nicht verfügbar:", e)
    if os.path.exists(path):
        g = load_boundary(path)
        x0, x1, y0, y1 = g.GetEnvelope()
        return g, (x0, y0, x1, y1)
    W, S, E, N = map(float, a.bbox.split(","))
    tr = osr.CoordinateTransformation(srs(4326), srs(2056))
    pts = [tr.TransformPoint(x, y) for x in (W, E) for y in (S, N)]
    return None, (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def load_boundary(path):
    ds = ogr.Open(path)
    lyr = ds.GetLayer(0)
    g = None
    for f in lyr:
        gg = f.GetGeometryRef().Clone()
        g = gg if g is None else g.Union(gg)
    return g


def srs(epsg):
    s = osr.SpatialReference()
    s.ImportFromEPSG(epsg)
    s.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return s


def grid_tiles(bb, margin):
    """10-km-Kacheln (linke untere Ecke in m), die bb + margin abdecken."""
    x0, y0, x1, y1 = bb[0] - margin, bb[1] - margin, bb[2] + margin, bb[3] + margin
    return [(x, y) for x in range(int(math.floor(x0 / TILE) * TILE), int(x1), TILE)
            for y in range(int(math.floor(y0 / TILE) * TILE), int(y1), TILE)]


def alti_index(a, bb):
    path = os.path.join(a.cache, f"alti_index_{a.name}.json")
    if os.path.exists(path):
        return json.load(open(path))
    tr = osr.CoordinateTransformation(srs(2056), srs(4326))
    m = a.margin_km * 1000
    p0 = tr.TransformPoint(bb[0] - m, bb[1] - m)
    p1 = tr.TransformPoint(bb[2] + m, bb[3] + m)
    url = f"{STAC}?bbox={p0[0]:.5f},{p0[1]:.5f},{p1[0]:.5f},{p1[1]:.5f}&limit=100"
    best = {}
    n = 0
    while url:
        js = json.loads(http_get(url))
        for it in js.get("features", []):
            for asset in it.get("assets", {}).values():
                href = asset.get("href", "")
                mm = re.search(r"swissalti3d_(\d{4})_(\d{4})-(\d{4})_2_2056_5728\.tif$", href)
                if mm:
                    y, e, nn = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
                    key = f"{e}-{nn}"
                    if key not in best or best[key][0] < y:
                        best[key] = [y, href]
        n += len(js.get("features", []))
        url = next((l["href"] for l in js.get("links", []) if l.get("rel") == "next"), None)
        if n % 1000 < 100:
            log(f"  STAC: {n} Einträge, {len(best)} Kacheln")
    if not best:
        raise RuntimeError("STAC lieferte keine swissALTI3D-Kacheln")
    with open(path, "w") as f:
        json.dump(best, f)
    log(f"swissALTI3D: {len(best)} Kacheln à 1 km (Jahrgänge {sorted({v[0] for v in best.values()})})")
    return best


def build_dtm_tile(a, x, y, items):
    out = os.path.join(a.cache, f"dtm2_{x // 1000}_{y // 1000}.tif")
    if os.path.exists(out):
        return "cache"
    mine = [(k, v) for k, v in items.items()
            if x <= int(k.split("-")[0]) * 1000 < x + TILE and y <= int(k.split("-")[1]) * 1000 < y + TILE]
    if not mine:
        return "leer"
    tmp = tempfile.mkdtemp(dir=a.cache, prefix="tmp_dtm_")
    try:
        def dl(kv):
            k, (yr, href) = kv
            p = os.path.join(tmp, os.path.basename(href))
            try:
                return http_get(href, p)
            except urllib.error.HTTPError as e:
                log(f"  fehlt: {href} ({e.code})")
                return None
        with ThreadPoolExecutor(a.threads) as ex:
            files = list(ex.map(dl, mine))
        if any(f is None for f in files):
            raise RuntimeError(f"Kachel {x // 1000}/{y // 1000}: Downloads fehlgeschlagen, wird beim nächsten Lauf nochmals versucht")
        vrt = gdal.BuildVRT(os.path.join(tmp, "t.vrt"), files, resolution="user", xRes=2, yRes=2, srcNodata=-9999,
                            VRTNodata=-9999)
        vrt = None
        gdal.Translate(out + ".part", os.path.join(tmp, "t.vrt"), format="GTiff", projWin=[x, y + TILE, x + TILE, y],
                       outputType=gdal.GDT_Float32, noData=-9999,
                       creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "ZLEVEL=6"])
        os.replace(out + ".part", out)
        return f"{len(files)} geladen"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def veg_from_release(a, tiles):
    base = VHM_RELEASE.format(name=a.name)
    idx_path = os.path.join(a.cache, f"veg_index_{a.name}.csv")
    if not os.path.exists(idx_path):
        try:
            http_get(base + "index.csv", idx_path)
        except Exception as e:
            log("Vegetationshöhe: kein Release ch-vhm-%s (%s), nehme Originaldaten" % (a.name, e))
            return False
    want = []
    for r in csv.DictReader(open(idx_path)):
        bx0, by0, bx1, by1 = (float(r[k]) for k in ("minx", "miny", "maxx", "maxy"))
        if any(bx0 < x + TILE and bx1 > x and by0 < y + TILE and by1 > y for x, y in tiles):
            want.append(r["file"])
    todo = [f for f in want if not os.path.exists(os.path.join(a.cache, "veg_" + f))]
    log(f"Vegetationshöhe (Release): {len(want)} Kacheln, davon {len(todo)} zu laden")
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(lambda f: http_get(base + f, os.path.join(a.cache, "veg_" + f)), todo))
    return True


def veg_from_cog(a, tiles):
    """Fallback: Vegetationshöhenmodell LFI direkt lesen, 2 m (Mittel), je Pixel neuester Jahrgang."""
    years = [y for y in VHM_YEARS if _exists(VHM_COG.format(y=y))]
    log("Vegetationshöhe (LFI-COG) Jahrgänge:", years)
    n = TILE // 2
    for x, y in tiles:
        out = os.path.join(a.cache, f"veg_vhm2_{x // 1000}_{y // 1000}.tif")
        if os.path.exists(out):
            continue
        res = np.full((n, n), 255, np.uint8)
        for yr in years:
            ds = gdal.Warp("", VHM_COG.format(y=yr), format="MEM", outputBounds=[x, y, x + TILE, y + TILE], xRes=2,
                           yRes=2, resampleAlg="average", dstNodata=-9999, outputType=gdal.GDT_Float32,
                           multithread=True)
            v = ds.GetRasterBand(1).ReadAsArray()
            ok = (v > -50) & (v < 200) & (res == 255)
            res[ok] = np.clip(np.rint(np.maximum(v[ok], 0) * 2), 0, 254)
            if (res != 255).all():
                break
        if (res == 255).all():
            res[:] = 255
        d = gdal.GetDriverByName("GTiff").Create(out + ".part", n, n, 1, gdal.GDT_Byte,
                                                 ["COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"])
        d.SetGeoTransform((x, 2, 0, y + TILE, 0, -2))
        d.SetSpatialRef(srs(2056))
        d.GetRasterBand(1).SetNoDataValue(255)
        d.GetRasterBand(1).WriteArray(res)
        d = None
        os.replace(out + ".part", out)
        log(f"  Vegetation {x // 1000}/{y // 1000} fertig")


def _exists(path):
    try:
        gdal.Open(path)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="thurgau")
    ap.add_argument("--canton", type=int, default=20, help="Kantonsnummer (BFS), 0 = nur --bbox")
    ap.add_argument("--bbox", default="8.65,47.36,9.52,47.71", help="W,S,E,N in Grad, falls keine Kantonsgrenze")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--margin-km", type=float, default=2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-veg", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    for d in os.listdir(a.cache):
        if d.startswith("tmp_"):
            shutil.rmtree(os.path.join(a.cache, d), ignore_errors=True)

    _, bb = boundary(a)
    tiles = grid_tiles(bb, a.margin_km * 1000)
    log(f"Gebiet {a.name}: LV95 {bb[0]:.0f}/{bb[1]:.0f} - {bb[2]:.0f}/{bb[3]:.0f}, {len(tiles)} Kacheln à 10 km")

    items = alti_index(a, bb)
    failed = 0
    for k, (x, y) in enumerate(tiles, 1):
        try:
            r = build_dtm_tile(a, x, y, items)
        except Exception as e:
            r = f"FEHLER {e}"
            failed += 1
        log(f"  Höhenmodell {k}/{len(tiles)} {x // 1000}/{y // 1000}: {r}")

    if not a.no_veg:
        if not veg_from_release(a, tiles):
            veg_from_cog(a, tiles)
    if failed:
        log(f"{failed} Kacheln unvollständig")
        sys.exit(1)
    log("Eingangsdaten vollständig im Cache")


if __name__ == "__main__":
    main()
