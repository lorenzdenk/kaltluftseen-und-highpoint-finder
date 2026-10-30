# Kaltluftseen- und Highpoint-Finder

Sucht im Höhenmodell **swissALTI3D** (2 m) Orte, an denen sich in klaren, windstillen Nächten Kaltluft sammelt, und gibt
sie als **GPX** aus (dazu GeoJSON und CSV). Standardgebiet: Kanton Thurgau.

## Arten von Kaltluftseen

| Typ | Was | Wie gerechnet |
|---|---|---|
| **Senke** | geschlossene Mulde | Priority-Flood-Füllung des Geländes |
| **Vegetation** | Wald, Waldrand oder Hecke staut den Abfluss | Vegetation als teildurchlässiger Damm: Gelände + min(0,4 × Vegetationshöhe, 6 m), ab 3 m Vegetationshöhe |
| **Talverengung** | Kaltluft fliesst ab, aber eine Engstelle lässt nicht alles durch | siehe unten |

**Talverengung:** Alle ~50 m entlang der Tiefenlinien (ab 0,1 km² Einzugsgebiet) wird ein Querschnitt quer zum Tal
gelegt. Gesucht ist die Kaltluftmächtigkeit *h*, bei der der kritische Abfluss Q = √(g′·A³/B) gleich der im
Einzugsgebiet erzeugten Kaltluft ist (12 m³ je m² und Stunde, Einzugsgebiet höchstens 20 km², g′ aus 3 K
Temperaturdefizit). Jeder Querschnitt wird ein virtueller Damm auf Höhe Tiefenlinie + *h*, danach wird wieder gefüllt.
Ein See zählt nur, wenn er weiter oben liegende Querschnitte desselben Tals um mindestens 1 m über deren eigenes
Fliessniveau hinaus überstaut und das Tal dort auf Seespiegelhöhe mindestens doppelt so breit ist wie die Engstelle.
Das sind grobe Annahmen, die Parameter lassen sich anpassen (`python3 scripts/kaltluftseen.py -h`).

Nicht berücksichtigt: Gebäude und Lärmschutzwände (nicht im Geländemodell), Wind, Durchlässe unter Dämmen
(ein Bahndamm mit Bachdurchlass gilt als dicht, für Kaltluft meist zutreffend).

## Ausgabe (Release `kaltluftseen-<name>`)

- `kaltluftseen_<name>.gpx`: ein Wegpunkt je See an der tiefsten Stelle. Name `KLS <Rang> <Volumen>`, sortiert nach
  Volumen. Die Beschreibung enthält Typ, Grund des Staus, Fläche, maximale und mittlere Tiefe, Seespiegel und Abflusspunkt.
- `kaltluftseen_<name>_umrisse.gpx`: Umrisse als Tracks.
- `kaltluftseen_<name>.geojson`, `.csv`: alle Kennzahlen.

Volumen ist die beste Einzelzahl (Menge der gestauten Kaltluft). Für Frostgefahr an einem Punkt zählt eher die Tiefe.

## Ablauf

```
python3 scripts/fetch.py --name thurgau --canton 20 --cache cache    # lädt nur, was im Cache fehlt
python3 scripts/kaltluftseen.py --name thurgau --cache cache --out out
```

Benötigt GDAL (Python), NumPy, SciPy, Numba. Rechenraster 5 m (`--res`), ganzer Thurgau braucht ~10 GB RAM.

**Cache:** `fetch.py` legt alles in `cache/` ab (swissALTI3D als 10-km-Kacheln, Vegetationshöhe, Kantonsgrenze) und lädt
beim nächsten Lauf nur Fehlendes. Auf GitHub sichert der Workflow den Cache im Release `cache-<name>` und holt ihn von
dort zurück, die Originaldaten werden also nur einmal geladen.

**GitHub:** Workflow „Kaltluftseen“ (startet bei Änderungen an `scripts/` oder manuell, mit Gebiet, Kantonsnummer und
zusätzlichen Parametern).

## Daten

- swissALTI3D © swisstopo (Open Government Data)
- Vegetationshöhenmodell LiDAR LFI, BAFU/WSL (Landesforstinventar, aus swissSURFACE3D). Übernommen aus dem Release
  `ch-vhm-<name>` von [render-baumhoehen-overlay](https://github.com/lorenzdenk/render-baumhoehen-overlay), sonst
  direkt von data.geo.admin.ch.
- Kantonsgrenze: swissBOUNDARIES3D © swisstopo
