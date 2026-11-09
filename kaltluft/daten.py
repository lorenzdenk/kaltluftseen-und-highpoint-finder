"""Eingangsdaten für ein Gebiet laden und im Cache ablegen. Was schon da ist, wird nicht nochmals geladen.

Cache-Inhalt (flach, damit er 1:1 als GitHub-Release abgelegt werden kann; Kacheln 10 x 10 km, E/N = linke untere
Ecke in km, LV95):
  boundary_<name>.geojson   Kantonsgrenze (swissBOUNDARIES3D via api3.geo.admin.ch)
  alti_index_<name>.json    Liste der swissALTI3D-Kacheln (STAC), je 1-km-Kachel neuester Jahrgang
  dtm2_<E>_<N>.tif          swissALTI3D 2 m, Float32, NoData -9999
  veg_<...>.tif             Vegetationshöhe (Byte, Wert = Höhe * 2, 255 = keine Daten)
  lc_<E>_<N>.tif            Landbedeckung ESA WorldCover 10 m (Byte, Klassen 10..100, 0 = keine Daten)

Vegetationshöhe: zuerst aus dem Release "ch-vhm-<name>" des Repos render-baumhoehen-overlay (1 m, schon aufbereitet),
sonst direkt aus dem Vegetationshöhenmodell LiDAR LFI (BAFU/WSL), je Pixel neuester Jahrgang, 2 m.
"""
import csv
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import shape
from shapely.ops import unary_union

from .raster import LV95, from_wgs84, log, to_wgs84

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"
BOUNDARY = ("https://api3.geo.admin.ch/rest/services/api/MapServer/ch.swisstopo.swissboundaries3d-kanton-flaeche.fill/"
            "{canton}?geometryFormat=geojson&sr=2056&returnGeometry=true")
VHM_RELEASE = "https://github.com/lorenzdenk/render-baumhoehen-overlay/releases/download/ch-vhm-{name}/"
VHM_COG = ("https://data.geo.admin.ch/ch.bafu.landesforstinventar-vegetationshoehenmodell_lidar/"
           "landesforstinventar-vegetationshoehenmodell_lidar_{y}/landesforstinventar-vegetationshoehenmodell_lidar_{y}_2056.tif")
WORLDCOVER = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_{lat}{lon}_Map.tif")
TILE = 10000
UA = {"User-Agent": "kaltluftseen-finder/2.0"}
GTIFF = dict(driver="GTiff", compress="deflate", tiled=True, blockxsize=512, blockysize=512, crs=LV95, count=1)


def http_get(url, pfad=None, tries=5):
    err = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
            if pfad is None:
                return data
            with open(pfad + ".part", "wb") as f:
                f.write(data)
            os.replace(pfad + ".part", pfad)
            return pfad
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            err = e
        except Exception as e:  # Netzfehler: nochmals
            err = e
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"{url}: {err}")


class _HttpDatei:
    """Datei über HTTP-Range-Anfragen lesen (für Cloud-optimierte GeoTIFF), mit Block-Cache. So läuft der Zugriff über
    urllib und ist unabhängig davon, wie GDAL/curl in der jeweiligen Installation für HTTPS eingerichtet ist."""
    BLOCK = 1 << 20

    def __init__(self, url, mode="rb", **kw):
        if not url.startswith("http"):
            raise FileNotFoundError(url)
        self.url = url
        self.pos = 0
        self.blocks = {}
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def _block(self, k):
        if k not in self.blocks:
            a, b = k * self.BLOCK, min((k + 1) * self.BLOCK, self.size) - 1
            err = None
            for n in range(5):
                try:
                    req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={a}-{b}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        self.blocks[k] = r.read()
                    break
                except Exception as e:
                    err = e
                    time.sleep(2 * (n + 1))
            else:
                raise RuntimeError(f"{self.url}: {err}")
            if len(self.blocks) > 256:
                self.blocks.pop(next(iter(self.blocks)))
        return self.blocks[k]

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            k, off = divmod(self.pos, self.BLOCK)
            teil = self._block(k)[off:off + end - self.pos]
            out += teil
            self.pos += len(teil)
        return bytes(out)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def close(self):
        self.blocks = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    closed = False


def oeffne_cog(url):
    return rasterio.open(url, opener=_HttpDatei)


def lies_grenze(pfad):
    """Gebietsgrenze (shapely, LV95) aus dem Cache."""
    with open(pfad) as f:
        fc = json.load(f)
    return unary_union([shape(ft["geometry"]) for ft in fc["features"]])


def grenze(cache, name, canton, bbox):
    """Kantonsgrenze laden (gecacht). Rückgabe: bbox in LV95 (x0, y0, x1, y1)."""
    pfad = os.path.join(cache, f"boundary_{name}.geojson")
    if not os.path.exists(pfad) and canton:
        try:
            feat = json.loads(http_get(BOUNDARY.format(canton=canton)))["feature"]
            fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": LV95}},
                  "features": [{"type": "Feature", "properties": feat.get("properties", {}),
                                "geometry": feat["geometry"]}]}
            with open(pfad, "w") as f:
                json.dump(fc, f)
            log("Kantonsgrenze geladen:", feat.get("properties", {}).get("name", canton))
        except Exception as e:
            log("Kantonsgrenze nicht verfügbar:", e)
    if os.path.exists(pfad):
        return lies_grenze(pfad).bounds
    w, s, e, n = map(float, bbox.split(","))
    pts = [from_wgs84(x, y) for x in (w, e) for y in (s, n)]
    return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)


def kacheln(bb, rand):
    """10-km-Kacheln (linke untere Ecke in m), die bb + rand abdecken."""
    x0, y0, x1, y1 = bb[0] - rand, bb[1] - rand, bb[2] + rand, bb[3] + rand
    return [(x, y) for x in range(int(math.floor(x0 / TILE) * TILE), int(math.ceil(x1)), TILE)
            for y in range(int(math.floor(y0 / TILE) * TILE), int(math.ceil(y1)), TILE)]


# ---------------------------------------------------------------- Höhenmodell ----

def alti_index(cache, name, bb, rand):
    pfad = os.path.join(cache, f"alti_index_{name}.json")
    if os.path.exists(pfad):
        with open(pfad) as f:
            return json.load(f)
    p0 = to_wgs84(bb[0] - rand, bb[1] - rand)
    p1 = to_wgs84(bb[2] + rand, bb[3] + rand)
    url = f"{STAC}?bbox={p0[0] - 0.01:.5f},{p0[1] - 0.01:.5f},{p1[0] + 0.01:.5f},{p1[1] + 0.01:.5f}&limit=100"
    best = {}
    while url:
        js = json.loads(http_get(url))
        for it in js.get("features", []):
            for asset in it.get("assets", {}).values():
                href = asset.get("href", "")
                mm = re.search(r"swissalti3d_(\d{4})_(\d{4})-(\d{4})_2_2056_5728\.tif$", href)
                if mm:
                    y, e, n = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
                    key = f"{e}-{n}"
                    if key not in best or best[key][0] < y:
                        best[key] = [y, href]
        url = next((l["href"] for l in js.get("links", []) if l.get("rel") == "next"), None)
    if not best:
        raise RuntimeError("STAC lieferte keine swissALTI3D-Kacheln")
    with open(pfad, "w") as f:
        json.dump(best, f)
    log(f"swissALTI3D: {len(best)} Kacheln à 1 km (Jahrgänge {sorted({v[0] for v in best.values()})})")
    return best


def dtm_kachel(cache, x, y, items, threads):
    out = os.path.join(cache, f"dtm2_{x // 1000}_{y // 1000}.tif")
    if os.path.exists(out):
        return "cache"
    mine = [v[1] for k, v in items.items()
            if x <= int(k.split("-")[0]) * 1000 < x + TILE and y <= int(k.split("-")[1]) * 1000 < y + TILE]
    if not mine:
        return "leer"
    tmp = tempfile.mkdtemp(dir=cache, prefix="tmp_dtm_")
    try:
        def dl(href):
            try:
                return http_get(href, os.path.join(tmp, os.path.basename(href)))
            except urllib.error.HTTPError as e:
                log(f"  fehlt: {href} ({e.code})")
                return None
        with ThreadPoolExecutor(threads) as ex:
            files = list(ex.map(dl, mine))
        if any(f is None for f in files):
            raise RuntimeError("Downloads fehlgeschlagen, wird beim nächsten Lauf nochmals versucht")
        arr, _ = merge(files, bounds=(x, y, x + TILE, y + TILE), res=2, nodata=-9999, dtype="float32")
        with rasterio.open(out + ".part", "w", height=arr.shape[1], width=arr.shape[2], dtype="float32", nodata=-9999,
                           transform=from_origin(x, y + TILE, 2, 2), predictor=3, zlevel=6, **GTIFF) as dst:
            dst.write(arr[0], 1)
        os.replace(out + ".part", out)
        return f"{len(files)} geladen"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- Vegetationshöhe ----

def veg_aus_release(cache, name, tiles, threads):
    base = VHM_RELEASE.format(name=name)
    idx = os.path.join(cache, f"veg_index_{name}.csv")
    if not os.path.exists(idx):
        try:
            http_get(base + "index.csv", idx)
        except Exception as e:
            log(f"Vegetationshöhe: kein Release ch-vhm-{name} ({e}), nehme Originaldaten")
            return False
    want = []
    with open(idx) as f:
        for r in csv.DictReader(f):
            bx0, by0, bx1, by1 = (float(r[k]) for k in ("minx", "miny", "maxx", "maxy"))
            if any(bx0 < x + TILE and bx1 > x and by0 < y + TILE and by1 > y for x, y in tiles):
                want.append(r["file"])
    todo = [f for f in want if not os.path.exists(os.path.join(cache, "veg_" + f))]
    log(f"Vegetationshöhe (Release): {len(want)} Kacheln, davon {len(todo)} zu laden")
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda f: http_get(base + f, os.path.join(cache, "veg_" + f)), todo))
    return True


def _gibt_es(url):
    try:
        with oeffne_cog(url):
            return True
    except Exception:
        return False


def veg_aus_cog(cache, tiles):
    """Vegetationshöhenmodell LFI direkt lesen, 2 m (Mittel), je Pixel neuester Jahrgang."""
    todo = [(x, y) for x, y in tiles if not os.path.exists(os.path.join(cache, f"veg_vhm2_{x // 1000}_{y // 1000}.tif"))]
    if not todo:
        return
    jahre = [j for j in range(time.gmtime().tm_year, 2016, -1) if _gibt_es(VHM_COG.format(y=j))]
    log("Vegetationshöhe (LFI) Jahrgänge:", jahre)
    n = TILE // 2
    for x, y in todo:
        out = os.path.join(cache, f"veg_vhm2_{x // 1000}_{y // 1000}.tif")
        res = np.full((n, n), 255, np.uint8)
        tr = from_origin(x, y + TILE, 2, 2)
        for j in jahre:
            v = np.full((n, n), np.nan, np.float32)
            with oeffne_cog(VHM_COG.format(y=j)) as src:
                reproject(rasterio.band(src, 1), v, dst_transform=tr, dst_crs=LV95, dst_nodata=np.nan,
                          resampling=Resampling.average)
            ok = (v > -50) & (v < 200) & (res == 255)
            res[ok] = np.clip(np.rint(np.maximum(v[ok], 0) * 2), 0, 254)
            if (res != 255).all():
                break
        with rasterio.open(out + ".part", "w", height=n, width=n, dtype="uint8", nodata=255, transform=tr,
                           predictor=2, **GTIFF) as dst:
            dst.write(res, 1)
        os.replace(out + ".part", out)
        log(f"  Vegetation {x // 1000}/{y // 1000} fertig")


# ---------------------------------------------------------------- Landbedeckung ----

def landbedeckung(cache, tiles):
    """ESA WorldCover 2021 (10 m) auf LV95-Kacheln umrechnen. Fehlschlag ist nicht fatal (Modell rechnet dann ohne
    Siedlung und Wasser)."""
    n = TILE // 10
    fehlt = 0
    for x, y in tiles:
        out = os.path.join(cache, f"lc_{x // 1000}_{y // 1000}.tif")
        if os.path.exists(out):
            continue
        ecken = [to_wgs84(xx, yy) for xx in (x, x + TILE) for yy in (y, y + TILE)]
        lons = {int(math.floor(p[0] / 3) * 3) for p in ecken}
        lats = {int(math.floor(p[1] / 3) * 3) for p in ecken}
        res = np.zeros((n, n), np.uint8)
        tr = from_origin(x, y + TILE, 10, 10)
        try:
            for lo in sorted(lons):
                for la in sorted(lats):
                    url = WORLDCOVER.format(lat=f"{'N' if la >= 0 else 'S'}{abs(la):02d}",
                                            lon=f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}")
                    v = np.zeros((n, n), np.uint8)
                    with oeffne_cog(url) as src:
                        reproject(rasterio.band(src, 1), v, dst_transform=tr, dst_crs=LV95, src_nodata=0,
                                  dst_nodata=0, resampling=Resampling.nearest)
                    res = np.where(res == 0, v, res)
        except Exception as e:
            log(f"  Landbedeckung {x // 1000}/{y // 1000} nicht verfügbar: {e}")
            fehlt += 1
            continue
        with rasterio.open(out + ".part", "w", height=n, width=n, dtype="uint8", nodata=0, transform=tr, **GTIFF) as dst:
            dst.write(res, 1)
        os.replace(out + ".part", out)
        log(f"  Landbedeckung {x // 1000}/{y // 1000} fertig")
    return fehlt


def lade(name, cache="cache", canton=20, bbox="8.65,47.36,9.52,47.71", rand_km=2.0, threads=8, veg=True,
         lc=True, fenster=None):
    """Lädt alle fehlenden Eingangsdaten. fenster = (x0, y0, x1, y1) in LV95 beschränkt auf einen Ausschnitt.
    Rückgabe: Anzahl unvollständiger Höhenmodell-Kacheln."""
    os.makedirs(cache, exist_ok=True)
    for d in os.listdir(cache):
        if d.startswith("tmp_"):
            shutil.rmtree(os.path.join(cache, d), ignore_errors=True)
    bb = fenster or grenze(cache, name, canton, bbox)
    rand = rand_km * 1000
    tiles = kacheln(bb, rand)
    log(f"Gebiet {name}: LV95 {bb[0]:.0f}/{bb[1]:.0f} - {bb[2]:.0f}/{bb[3]:.0f}, {len(tiles)} Kacheln à 10 km")
    if all(os.path.exists(os.path.join(cache, f"dtm2_{x // 1000}_{y // 1000}.tif")) for x, y in tiles):
        log("Höhenmodell vollständig im Cache")
        fehler = 0
    else:
        # Index immer für ganze 10-km-Kacheln, damit keine halb gefüllte Kachel im Cache landet
        voll = (min(t[0] for t in tiles), min(t[1] for t in tiles),
                max(t[0] for t in tiles) + TILE, max(t[1] for t in tiles) + TILE)
        iname = name if fenster is None else f"{name}_{voll[0] // 1000}_{voll[1] // 1000}_{voll[2] // 1000}_{voll[3] // 1000}"
        items = alti_index(cache, iname, voll, 0)
        fehler = 0
        for k, (x, y) in enumerate(tiles, 1):
            try:
                r = dtm_kachel(cache, x, y, items, threads)
            except Exception as e:
                r = f"FEHLER {e}"
                fehler += 1
            log(f"  Höhenmodell {k}/{len(tiles)} {x // 1000}/{y // 1000}: {r}")
    if veg and not veg_aus_release(cache, name, tiles, threads):
        veg_aus_cog(cache, tiles)
    if lc and landbedeckung(cache, tiles):
        log("Landbedeckung unvollständig: das Modell rechnet dort ohne Siedlung und Wasser")
    log("Höhenmodell vollständig im Cache" if not fehler else f"{fehler} Höhenmodell-Kacheln unvollständig")
    return fehler
