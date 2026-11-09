# Kaltluftseen-Finder

Sucht Orte, an denen sich in klaren, windstillen Nächten Kaltluft sammelt. Grundlage ist das Höhenmodell
**swissALTI3D**, dazu Vegetationshöhe und Landbedeckung. Standardgebiet: Kanton Thurgau.

Zwei Verfahren laufen nacheinander und ergänzen sich:

| Verfahren | Raster | Findet | Art in der Ausgabe |
|---|---|---|---|
| **Senkenanalyse** | 5 m | geschlossene Mulden, Stau hinter Wald und Hecken | `Senke`, `Vegetation` |
| **Kaltluftabfluss-Modell** | 20 m | wohin die Kaltluft fliesst, wo sie sich staut und liegen bleibt | `Stau` |

## Das Modell

### Senken

Das Gelände wird mit Priority-Flood gefüllt. Jede Mulde, aus der Kaltluft nur über den tiefsten Randpunkt abfliessen
kann, ist eine Senke. Wald und Hecken wirken als teildurchlässiger Damm: wirksame Oberfläche = Gelände +
min(0,4 × Vegetationshöhe, 12 m), ab 3 m Vegetationshöhe. Läge der Spiegel ohne Bäume deutlich tiefer, heisst die Senke
`Vegetation`. Je Senke: Volumen, Tiefe, Fläche, Überlaufpunkt, Kaltluft-Einzugsgebiet und Füllzeit.

### Kaltluftabfluss

Eine Kaltluftschicht entsteht auf dem Gelände, fliesst hangab, sammelt sich in den Tälern und staut sich an
Engstellen. Gerechnet wird eine tiefengemittelte Schicht mit reduzierter Schwere g′ = g · ΔT / T
(Flachwassergleichungen ohne Advektionsterm, wie in Ein-Schicht-Kaltluftmodellen üblich):

```
∂h/∂t + div(q) = P                         h Schichtdicke, q Volumenstrom je m Breite
∂q/∂t = −g′ h ∇(z + h) − cd |q| q / h²
```

| Landnutzung | Produktion P (m³/m²h) | Widerstand |
|---|---|---|
| Freiland (Wiese, Acker) | 12 | cd = 0,012 |
| Wald | 8 | + 0,12 bis Baumhöhe, darüber abnehmend |
| Siedlung | 1 | + 0,10 bis 8 m Höhe |
| Wasser | 0 | wie Freiland |

Wald kommt aus der Vegetationshöhe, Siedlung und Wasser aus ESA WorldCover. Am Rand des Gebiets fliesst die Kaltluft
frei ab. Die Simulation startet ohne Kaltluft und wird zu mehreren Zeitpunkten ausgewertet (Standard 1, 2 und 4 h).
Im Testgebiet (Aachtal bei Amriswil) ist der Zustand nach rund 2 bis 3 Stunden nahezu stationär.

Ein **Kaltluftsee (`Stau`)** ist ein zusammenhängendes Gebiet, in dem die Schicht mindestens 5 m dick ist und
langsamer als 0,2 m/s fliesst (`--see-schicht`, `--see-tempo`). Dort bleibt die Luft liegen und kühlt weiter aus.
Dicke Schichten, die zügig fliessen, sind Kaltluftströme und keine Seen.

Für jede Senke steht zusätzlich in der Ausgabe, wie dick die Kaltluftschicht dort ist und wie schnell sie fliesst:
eine Senke unter einem kräftigen Kaltluftstrom wird durchlüftet, eine unter stehender Luft nicht.

### Grenzen

- Konstantes Temperaturdefizit (3 K): kein Wärmehaushalt, die Schichtdicke ist keine Temperatur.
- Kein übergeordneter Wind. Das Modell gilt für windschwache Strahlungsnächte.
- 20-m-Raster: schmale Dämme, Lärmschutzwände und einzelne Gebäude sind nicht aufgelöst. `--res-fluss 10` rechnet
  feiner (4-mal mehr Zellen, etwa 8-mal länger).
- Durchlässe unter Dämmen fehlen im Geländemodell, ein Bahndamm gilt als dicht.
- Nicht an Messungen geeicht. Die Parameter sind Literaturwerte (VDI 3787 Blatt 5) und lassen sich alle über die
  Kommandozeile ändern. Die Version 1 hatte eine Produktionsrate von 3 m³/m²h an einem bekannten Kaltluftsee bei
  Geboltshausen abgeglichen. Das galt für das damalige Verfahren (Talquerschnitte) und ist nicht übertragbar.

## Ausgabe

Dateien in `out/`, auf GitHub im Release `kaltluftseen-<name>`:

| Datei | Inhalt |
|---|---|
| `kaltluftseen_<name>.gpx` | Wegpunkt je Senke (tiefste Stelle) und je Kaltluftsee des Hauptzeitpunkts |
| `kaltluftseen_<name>_top200.gpx` | die 200 grössten je Art |
| `kaltluftseen_<name>_umrisse.gpx` | Umrisse als Tracks |
| `kaltluftseen_<name>.geojson`, `.csv` | alle Objekte mit allen Kennzahlen, Kaltluftseen aller Zeitpunkte (`nach_h`) |
| `kaltluft_<name>_<N>h.tif` | 4 Bänder: Schichtdicke (m), Fliessgeschwindigkeit (m/s), Volumenstrom (m²/s), Fliessrichtung (°) |
| `kaltluft_<name>_beginn.tif` | Stunden, bis die Schicht 5 m dick ist |
| `kaltluft_<name>_<N>h.png` | Übersichtskarte |
| `parameter_<name>.json` | verwendete Einstellungen |

Senken heissen `KLS <Rang> <Volumen>`, Kaltluftseen aus dem Strömungsmodell `Stau <Rang> <Fläche> <Dicke>`.

## Ablauf

```
pip install -e .
python -m kaltluft daten   --name thurgau --canton 20     # lädt nur, was im Cache fehlt
python -m kaltluft rechnen --name thurgau
```

Ausschnitt statt ganzem Kanton (LV95): `--fenster 2735000,1264000,2744000,1272000` bei beiden Befehlen. Alle
Einstellungen: `python -m kaltluft rechnen -h`. Nur Senken: `--ohne-stroemung`. Nur Geländemulden ohne Bäume:
`--veg-k 0`.

Ganzer Thurgau (Schätzung, hochgerechnet aus einem Ausschnitt von 13 × 12 km): Senken rund 5 Minuten und 9 GB RAM,
Strömungsmodell bei 20 m rund 10 bis 20 Minuten.

**Tests:** `pip install -e .[test] && pytest` (künstliches Gelände, ohne Netz).

**Cache:** `daten` legt alles in `cache/` ab (10-km-Kacheln) und lädt beim nächsten Lauf nur Fehlendes. Auf GitHub
sichert der Workflow den Cache im Release `cache-<name>`.

**GitHub:** Workflow „Kaltluftseen“ startet bei Änderungen am Code auf `main` oder manuell (Gebiet, Kantonsnummer,
zusätzliche Parameter).

## Aufbau

```
kaltluft/daten.py      Eingangsdaten laden
kaltluft/senken.py     Senkenanalyse (Priority-Flood)
kaltluft/stroemung.py  Kaltluftabfluss-Modell
kaltluft/seen.py       Kaltluftseen aus dem Strömungsfeld
kaltluft/ausgabe.py    GPX, GeoJSON, CSV, GeoTIFF, Karten
kaltluft/cli.py        Kommandozeile und Ablauf
scripts/ecostress_*.py ECOSTRESS-Nachtaufnahmen (Oberflächentemperatur) zum Vergleich suchen
```

## Daten

- swissALTI3D © swisstopo (Open Government Data)
- Vegetationshöhenmodell LiDAR LFI, BAFU/WSL. Übernommen aus dem Release `ch-vhm-<name>` von
  [render-baumhoehen-overlay](https://github.com/lorenzdenk/render-baumhoehen-overlay), sonst direkt von
  data.geo.admin.ch.
- ESA WorldCover 10 m 2021 v200 (CC BY 4.0)
- Kantonsgrenze: swissBOUNDARIES3D © swisstopo
