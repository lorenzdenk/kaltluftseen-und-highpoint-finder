"""Tests mit künstlichem Gelände (kein Netz nötig)."""
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from kaltluft import cli, seen, senken, stroemung
from kaltluft.raster import Gitter, block_mittel


def schuessel(n=120, dx=5.0, tiefe=4.0, radius=100.0):
    """Geneigte Ebene mit einer runden Mulde in der Mitte."""
    y, x = np.mgrid[0:n, 0:n] * dx
    d = np.hypot(x - n * dx / 2, y - n * dx / 2)
    return (500 + 0.01 * x - tiefe * np.clip(1 - (d / radius) ** 2, 0, None)).astype(np.float32)


def tal(n=200, dx=20.0, engstelle=True):
    """Nach Osten fallendes Tal; bei x = 2500 m verengt es sich stark."""
    y, x = np.mgrid[0:n, 0:n] * dx
    breite = 800 - (700 * np.exp(-((x - 2500) / 200) ** 2) if engstelle else 0)
    quer = np.abs(y - (n - 1) * dx / 2)  # Talachse genau in der Mitte des Rasters
    return (500 - 0.005 * x + 0.2 * np.clip(quer - breite / 2, 0, None)).astype(np.float32)


def test_gitter_und_blockmittel():
    g = Gitter((2700003, 1250001, 2700996, 1250499), 20)
    assert (g.x0, g.y0) == (2700000, 1250000)
    assert g.shape == (25, 50)
    assert g.rc(*g.xy(3, 7)) == (3, 7)
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    a[0, 0] = np.nan
    b = block_mittel(a, 2)
    assert b.shape == (2, 2)
    assert b[0, 0] == pytest.approx((1 + 4 + 5) / 3)
    assert b[1, 1] == pytest.approx((10 + 11 + 14 + 15) / 4)


def test_senke_wird_gefunden():
    z = schuessel()
    sen, lab = senken.finde_senken(z, np.zeros_like(z), 5.0, veg_k=0, min_flaeche_ha=0.1, min_volumen=100)
    assert len(sen) == 1
    s = sen[0]
    assert s["typ"] == "Senke"
    assert 2.0 < s["tiefe_max"] < 4.0  # die Neigung der Ebene senkt den Überlauf unter die volle Muldentiefe
    assert abs(s["r"] - 60) <= 12 and abs(s["c"] - 60) <= 12
    assert s["volumen"] == pytest.approx((lab == s["lab"]).sum() * 25 * s["tiefe_mittel"], rel=1e-3)
    assert s["einzugsgebiet_ha"] >= s["flaeche_ha"]


def test_hecke_staut():
    y, x = np.mgrid[0:120, 0:120] * 5.0
    z = (500 + 0.02 * x + 0.05 * np.abs(y - 300)).astype(np.float32)  # Tälchen, fällt nach Westen
    v = np.zeros_like(z)
    v[:, 20:23] = 15.0  # Hecke quer zum Tal
    ohne, _ = senken.finde_senken(z, v, 5.0, veg_k=0, min_flaeche_ha=0.05, min_volumen=100)
    mit, _ = senken.finde_senken(z, v, 5.0, veg_k=0.4, min_flaeche_ha=0.05, min_volumen=100)
    assert ohne == []
    assert len(mit) == 1 and mit[0]["typ"] == "Vegetation"
    assert mit[0]["stau_baeume"] > 2


def test_massenbilanz_im_geschlossenen_becken():
    n, dx = 80, 20.0
    y, x = np.mgrid[0:n, 0:n] * dx
    z = (400 + 0.0005 * ((x - 800) ** 2 + (y - 800) ** 2)).astype(np.float32)  # Rand 320 m über der Mitte
    r = stroemung.simuliere(z, dx, [0.5])
    b = r["bilanz"]
    # nur die Randzellen verlieren Kaltluft (freier Abfluss nach aussen): 5 % der Fläche
    assert b["vorhanden_m3"] == pytest.approx(b["erzeugt_m3"], rel=0.03)
    assert b["vorhanden_m3"] <= b["erzeugt_m3"]
    h = r["zustand"][0.5]["h"]
    assert np.isfinite(h).all() and h.min() >= 0
    assert h[40, 40] > 5 * h[5, 5]  # Kaltluft sammelt sich in der Mitte
    # Kaltluftoberfläche in der Mitte nahezu waagrecht
    ob = (z + h)[35:45, 35:45]
    assert ob.max() - ob.min() < 1.0


def test_symmetrie():
    z = tal()
    h = stroemung.simuliere(z, 20.0, [0.5])["zustand"][0.5]["h"]
    assert np.allclose(h, h[::-1], atol=0.02)


def test_engstelle_staut_und_wird_als_see_erkannt():
    dx = 20.0
    mit = stroemung.simuliere(tal(engstelle=True), dx, [2])["zustand"][2.0]
    ohne = stroemung.simuliere(tal(engstelle=False), dx, [2])["zustand"][2.0]
    oberhalb = (slice(95, 106), slice(80, 110))
    assert mit["h"][oberhalb].mean() > ohne["h"][oberhalb].mean() + 2  # Rückstau
    assert mit["tempo"][oberhalb].mean() < ohne["tempo"][oberhalb].mean()  # und langsamer
    assert mit["tempo"][100, 125] > 2 * mit["tempo"][100, 95]  # schnell durch die Engstelle
    # Fliessrichtung im Tal: nach Osten
    assert 60 < seen.richtung(mit["qx"], mit["qy"])[100, 150] < 120
    objs, lab = seen.finde_stauseen(mit, tal(), dx, min_h=5, max_tempo=0.2, min_flaeche_ha=1)
    assert any(o["c"] < 125 and abs(o["r"] - 100) < 25 for o in objs)


def test_wald_bremst_und_wasser_produziert_nichts():
    par = stroemung.Parameter()
    n = 50
    valid = np.ones((n, n), bool)
    wald = np.zeros((n, n), np.float32)
    wald[:, 25:] = 1
    wasser = np.zeros((n, n), np.float32)
    wasser[:5] = 1
    prod, cd0, cdo, hob = stroemung.felder(par, valid, wald, np.full((n, n), 20, np.float32), None, wasser)
    assert prod[0, 0] == 0
    assert prod[10, 0] == pytest.approx(par.prod_frei / 3600)
    assert prod[10, 30] == pytest.approx(par.prod_wald / 3600)
    assert cdo[10, 30] == pytest.approx(par.cd_wald) and cdo[10, 0] == 0
    y, x = np.mgrid[0:n, 0:n] * 20.0
    z = (500 - 0.05 * y).astype(np.float32)  # gleichmässiger Hang
    zu = stroemung.simuliere(z, 20.0, [0.5], par, prod, cd0, cdo, hob)["zustand"][0.5]
    assert zu["tempo"][30, 10] > 1.5 * zu["tempo"][30, 40]


def test_ganzer_lauf(tmp_path):
    """Kachel im Cache-Format schreiben, alles rechnen, Ausgabedateien prüfen."""
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    n = 1000  # 2 km bei 2 m
    y, x = np.mgrid[0:n, 0:n] * 2.0
    z = 500 - 0.004 * x + 0.15 * np.clip(np.abs(y - 1000) - 250, 0, None)
    z -= 5 * np.clip(1 - (np.hypot(x - 600, y - 1000) / 80) ** 2, 0, None)  # Mulde im Talboden
    z += 6 * np.exp(-((x - 1400) / 10) ** 2) * (np.abs(y - 1000) < 400)  # Damm quer durchs Tal
    with rasterio.open(cache / "dtm2_2700_1250.tif", "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:2056", transform=from_origin(2700000, 1252000, 2, 2), nodata=-9999) as dst:
        dst.write(z.astype(np.float32), 1)
    args = cli.parser().parse_args(["rechnen", "--name", "probe", "--cache", str(cache), "--out", str(out),
                                    "--stunden", "0.5,1", "--min-flaeche", "0.1", "--see-flaeche", "0.5"])
    objekte = cli.rechne(args)
    typen = {o["typ"] for o in objekte}
    assert "Senke" in typen
    for name in ("kaltluftseen_probe.gpx", "kaltluftseen_probe_umrisse.gpx", "kaltluftseen_probe.csv",
                 "kaltluftseen_probe.geojson", "kaltluft_probe_1h.tif", "kaltluft_probe_beginn.tif",
                 "parameter_probe.json"):
        assert (out / name).exists(), name
    gpx = ET.parse(out / "kaltluftseen_probe.gpx").getroot()
    wpt = gpx.findall("{http://www.topografix.com/GPX/1/1}wpt")
    assert len(wpt) >= 1
    assert 47.3 < float(wpt[0].get("lat")) < 47.5 and 8.6 < float(wpt[0].get("lon")) < 8.8
    fc = json.load(open(out / "kaltluftseen_probe.geojson"))
    assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon") for f in fc["features"])
    with rasterio.open(out / "kaltluft_probe_1h.tif") as src:
        assert src.count == 4 and src.res == (20, 20)
        h = src.read(1)
    assert np.nanmax(h) > 3
    senke = next(o for o in objekte if o["typ"] == "Senke")
    assert senke["schicht"] > 0 and senke["einzugsgebiet_ha"] > 0
