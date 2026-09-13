# Main Figure: SVG-Asset-Bibliothek

Startpunkt ist [`assets/index.html`](assets/index.html), die Gegenüberstellung
aller Bausteine. [`assets/preview.png`](assets/preview.png) zeigt die wichtigsten
Varianten auf einem Blatt. Die 21 SVGs sind die eigentlichen Arbeitsdateien:
transparenter Hintergrund, editierbare Kreise, Rechtecke, Linien und Texte,
keine eingebetteten Rasterbilder und keine externen Bild- oder Font-Abhängigkeiten.
Texte verwenden Arial mit Helvetica/sans-serif als Fallback.

## Erzeugen

Vom Repository-Root aus:

```powershell
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl"
```

Das Skript verwendet JLD2, JSON und Julia-Standardbibliotheken. Es liest die
vorhandenen Quelldaten und erzeugt die SVGs, HTML-Galerie und `provenance.json`
unter `Revision/Main Figure/assets`. Es startet keine Simulation, kein Training
und keine erneute Kandidatenauswahl. Beim bloßen `include` wird nichts erzeugt.

```powershell
# Nur Quelldaten und globale/lokale Maskenzuordnung prüfen:
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl" --check-only

# Drei benachbarte Agenten wie in der bisherigen Framework-Abbildung:
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl" --agents 5,6,7 --output-dir "Revision/Main Figure/adjacent_windows"

# Periodische Rand-Windows werden in zwei korrekt zugeordnete Teile zerlegt:
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl" --agents 1,6,12 --output-dir "Revision/Main Figure/periodic_windows"
```

Weitere Optionen: `--experiment-id ID` für eine andere bereits eingefrorene
Package-8-GO-GC-Auswahl und `--state-file PATH` für einen kompatiblen gespeicherten
Oceananigans-Two-Plume-Checkpoint. Das Standardexperiment ist bewusst fest auf
`260830_231109` gesetzt; neue Experimente ändern die Figure nicht automatisch.

Optional erzeugt `node "Revision/Main Figure/render_preview.cjs"` mit installiertem
`sharp` die PNG-Übersicht und Einzelansichten unter `assets/preview/`.
Ein abweichender Asset-Ordner kann als erstes Argument übergeben werden.
Die Julia-SVG-Erzeugung benötigt Node und `sharp` nicht.

## Bausteine

| Dateien in `assets/` | Verwendung |
| --- | --- |
| `dense_temperature.svg`, `sparse_temperature.svg` | Hauptassets: Temperatur an allen 48 × 8 Sensororten; inaktiv wird hellgrau |
| `dense_temperature_windows.svg`, `sparse_temperature_windows.svg` | Gleiche Sensorwerte mit drei echten Windows und allen zwölf Aktuatoren |
| `dense_local_windows.svg`, `sparse_local_windows.svg` | Drei separat herausgezogene 15 × 8 lokale Temperaturfenster |
| `dense_channel_mask.svg`, `sparse_channel_mask.svg` | Je Sensor drei Farbstreifen in der Reihenfolge T, w, u wie in den Maskenplots |
| `dense_channel_mask_windows.svg`, `sparse_channel_mask_windows.svg` | Kanalmaske mit drei Windows und Aktuatorleiste |
| `dense_vertical_velocity.svg`, `sparse_vertical_velocity.svg` | Tatsächlich gemessene vertikale Geschwindigkeit |
| `dense_horizontal_velocity.svg`, `sparse_horizontal_velocity.svg` | Tatsächlich gemessene horizontale Geschwindigkeit |
| `two_plume_temperature_field.svg` | Optionaler vollständiger 96 × 64 Temperaturzustand, ebenfalls rein vektorbasiert |
| `temperature_legend.svg`, `channel_legend.svg` | Separate Legenden für Temperaturwerte bzw. Kanalidentität |
| `actuators_12.svg` | Separate Leiste aller zwölf Aktuatorsegmente |
| `dense_controller.svg`, `sparse_controller.svg` | Zwei optionale Controller-Bausteine mit jeweils zwölf Agent-Tokens |
| `distillation_arrow.svg` | Separater Pfeil für Expert-Action-Targets und GO-GC-Regularisierung |

Die Kanalmaske verwendet kategorische Farben für **Kanalidentität**, nicht für
Messwerte. Die Temperatur- und Geschwindigkeitsdateien verwenden dagegen
kontinuierliche Farben für **physikalische Messwerte**.

## Konkrete Datenherkunft

- Zustand: `RBmodel300.jld2`, derselbe gespeicherte Two-Plume-Ausgangszustand,
  der in `Revision/Run_Files/FixedIC_MAT.jl` geladen wird. Die Figure kombiniert
  diesen illustrativen Zustand mit einer Varying-IC-Maske; sie behauptet damit
  keinen bestimmten Varying-IC-Testrollout.
- Maske: `Revision/Package8/results/260830_231109/go-gc/analysis/selected_test_candidate.jld2`.
- Kandidat: `2e7a2411c43d08d0c2041d86`, Run `p8_260830_231109_go_gc_s_0p02_r01`,
  Update `99850`, Regularisierungsstärke `0.02`, Maskenschwelle `0.003`.
- Auswahl: bereits vor dem Test eingefroren, ausschließlich Validation;
  Validation-MSE `0.00030600613603989284`. Das Skript prüft die gespeicherten
  Freeze- und Selection-Flags und übernimmt den Kandidaten unverändert.
- Aktiv: **1/32 GC-Gruppen, 12/384 Sensororte und 36/1152 skalare Messwerte**.
  Alle drei Kanäle haben dieselbe Ortsmaske. Aktiv sind die horizontalen
  Sensorindizes `1,5,9,...,45` in der vertikalen Sensorzeile `4` (von unten).
- `provenance.json` enthält SHA-256-Hashes des Zustands, der Auswahl und des
  Generators sowie den gespeicherten Checkpoint-Hash, Maskenindizes,
  Farbgrenzen und die Zuordnungen für alle zwölf Agenten.

## Wissenschaftliche Darstellung

Die physischen Proben werden exakt an den Feldindizes `x=1:2:95` und `z=1:8:57`
ausgelesen. Die Interior-Slices des Checkpoints entsprechen dem Run-File;
Kanalreihenfolge ist Temperatur `b`, vertikale Geschwindigkeit `w`, horizontale
Geschwindigkeit `u`. Es findet keine räumliche Interpolation statt.

Die globalen Glyphen stehen an den Temperatur-Zellzentren
`x=(ix-0.5)·2π/96`, `z=(iz-0.5)·2/64`; der physikalische Seitenquotient bleibt
erhalten. Da die Umgebung die gestaffelten Geschwindigkeitsfelder unter
denselben Indizes ausliest, werden die drei Kanäle hier gemeinsam an einem
Probe-Symbol dargestellt. Die unterschiedlichen nativen u/w-Positionen werden
nicht als zusätzliche Sensororte gezeichnet. Die unterste Probe ist deshalb
sehr nah an der unteren Wand, während die oberste Probe unterhalb der oberen
Wand liegt. Die lokalen Ausschnitte verwenden ein lesbares Indexraster.

Die Temperaturpalette und ihre festen Grenzen **[1, 2.5]** stammen aus dem
bisherigen RBC-Sensorplot. Dense und Sparse teilen dieselbe Skala. Auch die
beiden Geschwindigkeitskanäle teilen jeweils eine symmetrische Skala zwischen
Dense und Sparse; ihre Grenzen werden aus allen unmaskierten Sensorwerten
des geladenen Zustands bestimmt. Inaktive Glyphen erhalten exakt **#F2F2F2**
aus den Package-7/8-Maskenplots; die Kanalstreifen verwenden deren drei Farben.

Die Assets zeigen den physikalischen Zustand **vor dem sinusförmigen positional
encoding**. Der Temperaturkanal des echten MAT-Eingangs enthält zusätzlich
diesen bekannten Positionsanteil. Die Maske wird weder verschoben noch an den
Zustand angepasst. Die aktiven Punkte besitzen in beiden Varianten exakt
dieselben Farben und Messwerte.

Die zwölf Agentzentren sind die horizontalen Sensorindizes `3,7,...,47`.
Jedes Window umfasst modulo 48 genau sieben Spalten links, die Zentrumsspalte
und sieben Spalten rechts, jeweils alle acht Höhen und drei Kanäle:
**360 skalare Eingänge pro dichtem lokalen Window**. Die vorliegende Maske
behält darin vier Orte bzw. zwölf skalare Eingänge. Das Skript gleicht die
globale Maske für alle zwölf Agenten mit der gespeicherten lokalen Maske ab.

Standardmäßig sind Agenten **3, 6 und 9** mit leicht überlappenden Rahmen,
versetzten Klammern und passenden Aktuatorfarben dargestellt. Mit `--agents`
lässt sich die Auswahl ändern. Die Controller-Symbole stehen jeweils für
**einen gemeinsamen MAT mit zwölf Agent-Tokens**; der Transformer verarbeitet
die Beobachtungen gemeinsam. Sie stellen keine zwölf unabhängig
parametrisierten Netze dar.

Die finale Anordnung, Agent-Symbolik und der PDF-Export bleiben der manuellen
Figure-Komposition vorbehalten. Es werden hier ausschließlich SVG-Arbeitsassets
und deren HTML-/PNG-Vorschauen hergestellt.
