"""Rasterhilfen: Kacheln auf ein Rechenraster lesen, GeoTIFF schreiben, Umrisse bilden (alles LV95, EPSG:2056)."""
import math
import time

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

LV95 = "EPSG:2056"
_TO84 = Transformer.from_crs(LV95, "EPSG:4326", always_xy=True)
_FROM84 = Transformer.from_crs("EPSG:4326", LV95, always_xy=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def to_wgs84(x, y):
    return _TO84.transform(x, y)


def from_wgs84(lon, lat):
    return _FROM84.transform(lon, lat)


def geom_to_wgs84(geom):
    return shapely.transform(geom, lambda xy: np.column_stack(_TO84.transform(xy[:, 0], xy[:, 1])))


class Gitter:
    """Nordorientiertes Raster in LV95: bounds = (x0, y0, x1, y1), Zellgrösse res."""

    def __init__(self, bounds, res):
        x0, y0, x1, y1 = bounds
        self.res = float(res)
        self.x0 = math.floor(x0 / res) * res
        self.y0 = math.floor(y0 / res) * res
        self.nc = int(math.ceil((x1 - self.x0) / res))
        self.nr = int(math.ceil((y1 - self.y0) / res))
        self.x1 = self.x0 + self.nc * res
        self.y1 = self.y0 + self.nr * res

    @property
    def shape(self):
        return self.nr, self.nc

    @property
    def transform(self):
        return from_origin(self.x0, self.y1, self.res, self.res)

    def xy(self, r, c):
        """Zellmitte in LV95."""
        return self.x0 + (c + 0.5) * self.res, self.y1 - (r + 0.5) * self.res

    def rc(self, x, y):
        return int((self.y1 - y) // self.res), int((x - self.x0) // self.res)

    def grob(self, faktor):
        """Raster mit faktor-mal grösseren Zellen, das dieses Raster überdeckt (Zeilen/Spalten ggf. abgeschnitten)."""
        g = Gitter.__new__(Gitter)
        g.res = self.res * faktor
        g.x0, g.y1 = self.x0, self.y1
        g.nr, g.nc = self.nr // faktor, self.nc // faktor
        g.x1 = g.x0 + g.nc * g.res
        g.y0 = g.y1 - g.nr * g.res
        return g


def lies_mosaik(dateien, gitter, resampling="average", nodata=None):
    """Liest Kacheln auf das Raster (Float32, NaN = keine Daten). Erste Kachel mit Daten gewinnt."""
    out = np.full(gitter.shape, np.nan, np.float32)
    res = gitter.res
    rs = Resampling[resampling]
    for f in dateien:
        with rasterio.open(f) as src:
            b = src.bounds
            if b.right <= gitter.x0 or b.left >= gitter.x1 or b.top <= gitter.y0 or b.bottom >= gitter.y1:
                continue
            c0 = max(0, int(math.floor((b.left - gitter.x0) / res)))
            c1 = min(gitter.nc, int(math.ceil((b.right - gitter.x0) / res)))
            r0 = max(0, int(math.floor((gitter.y1 - b.top) / res)))
            r1 = min(gitter.nr, int(math.ceil((gitter.y1 - b.bottom) / res)))
            if c1 <= c0 or r1 <= r0:
                continue
            dst = np.full((r1 - r0, c1 - c0), np.nan, np.float32)
            reproject(rasterio.band(src, 1), dst, dst_transform=from_origin(gitter.x0 + c0 * res, gitter.y1 - r0 * res,
                                                                           res, res),
                      dst_crs=src.crs or LV95, src_nodata=nodata if nodata is not None else src.nodata,
                      dst_nodata=np.nan, resampling=rs, num_threads=4)
            sub = out[r0:r1, c0:c1]
            m = np.isnan(sub) & np.isfinite(dst)
            sub[m] = dst[m]
    return out


def block_mittel(a, f):
    """Mittel über f x f Blöcke, NaN werden ignoriert (Block ganz ohne Daten = NaN)."""
    nr, nc = a.shape[0] // f, a.shape[1] // f
    b = a[:nr * f, :nc * f].reshape(nr, f, nc, f)
    ok = np.isfinite(b)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0).sum(axis=(1, 3), dtype=np.float64)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def schreibe_tif(pfad, gitter, baender, namen=None, nodata=np.nan):
    """Schreibt ein oder mehrere Bänder (Liste von 2D-Arrays) als komprimiertes GeoTIFF."""
    baender = [np.asarray(b) for b in baender]
    dtype = baender[0].dtype
    prof = dict(driver="GTiff", height=gitter.nr, width=gitter.nc, count=len(baender), dtype=dtype, crs=LV95,
                transform=gitter.transform, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                nodata=nodata)
    if np.issubdtype(dtype, np.floating):
        prof["predictor"] = 3
    with rasterio.open(pfad, "w", **prof) as dst:
        for k, b in enumerate(baender, 1):
            dst.write(b, k)
            if namen:
                dst.set_band_description(k, namen[k - 1])


def umrisse(label, ids, gitter, vereinfachen=None):
    """Umrisse je Label-ID (LV95, shapely). label: 2D int32, < 0 = nichts."""
    ids = np.asarray(ids)
    if ids.size == 0:
        return {}
    g = np.where(np.isin(label, ids), label, -1).astype(np.int32)
    teile = {}
    for geom, val in features.shapes(g, mask=g >= 0, connectivity=8, transform=gitter.transform):
        teile.setdefault(int(val), []).append(shape(geom))
    tol = gitter.res * 0.75 if vereinfachen is None else vereinfachen
    out = {}
    for k, v in teile.items():
        u = unary_union(v) if len(v) > 1 else v[0]
        out[k] = u.simplify(tol, preserve_topology=True) if tol > 0 else u
    return out
