# Main Figure: Iteration 2

Die aktuelle Bibliothek enthält **56 editierbare SVGs**. Einstieg:
[filterbare Galerie](assets/index.html), [Übersicht](assets/preview.png),
[alle zwölf Windows im Vergleich](assets/all_windows_comparison.png) und
[vier Auswahlen mit drei Agenten](assets/three_agent_comparison.png).

**Ein Punkt steht für einen Sensorort mit allen drei Kanälen T, w und u.**
Nur seine Temperatur wird farblich visualisiert. Das gilt auf der Dense- und
Sparse-Seite gleichermaßen, da der GO-GC-Kandidat die Kanäle gemeinsam auswählt.
Die aktuelle Bibliothek enthält ausschließlich Temperaturansichten.
Die [erste Iteration](iterations/01/assets/index.html) ist separat archiviert.

## Farben und Window-Variationen

Dense-Agenten verwenden Lila, Sparse-Agenten Magenta. Die Agentnummer legt die
Farbe fest: Agent 1 ist am dunkelsten, Agent 12 am hellsten. Dadurch hat etwa
Agent 6 innerhalb jeder Farbfamilie in allen Varianten dieselbe Farbe.
Die Zuordnung gilt für Rahmen, Klammern, Agent-Labels, Aktuatoren und
Controller-Tokens. Die physischen Temperaturfarben ändern sich dadurch nicht.

| Auswahl / Dateikennung | Agenten | Visueller Schwerpunkt |
| --- | --- | --- |
| 03_06_09 | 3, 6, 9 | Bisherige, leicht überlappende Auswahl |
| 06 | 6 | Ein einzelnes Window |
| 04_08 | 4, 8 | Zwei getrennte Windows |
| 05_06_07 | 5, 6, 7 | Drei benachbarte, stark überlappende Windows |
| 02_06_10 | 2, 6, 10 | Drei weit verteilte Windows |
| 01_06_12 | 1, 6, 12 | Periodische Randübergänge |
| 02_05_08_11 | 2, 5, 8, 11 | Vier verteilte Windows |
| 04_05_06_07 | 4, 5, 6, 7 | Vier benachbarte Windows |
| 01_03_05_07_09_11 | 1, 3, 5, 7, 9, 11 | Jeder zweite Agent |
| all_12 | 1 bis 12 | Vollständiges Multi-Agent-Setup |

Jede Auswahl gibt es auf beiden Seiten in zwei Stilen:

- **frames:** gestrichelte Window-Rahmen, Klammern darüber und Agenten darunter.
- **brackets:** gleiche Klammern und Agenten, ohne Rahmen durch das Sensorfeld.

Bei bis zu vier Agenten bleibt die versetzte Klammeranordnung der ersten
Iteration erhalten. Größere Auswahlen teilen sich Zeilen ausschließlich bei
disjunkten Windows. Alle zwölf Agenten benötigen so vier Klammerzeilen.
Periodische Randstücke sind zusätzlich mit einem Pfeil und ihrer Agentnummer
markiert. Die Rahmen erhalten keine Flächenfüllung; die Sensorfarben bleiben
unverändert. Alle zwölf Aktuatorsegmente sind in jeder Auswahl sichtbar.

## Dateien

Die 40 Window-SVGs liegen unter assets/windows/ nach diesem Muster:

    dense_03_06_09_frames.svg
    sparse_03_06_09_frames.svg
    dense_all_12_brackets.svg
    sparse_all_12_brackets.svg

Weitere 16 SVGs liegen direkt unter assets/:

| Dateinamen | Inhalt |
| --- | --- |
| dense_temperature.svg / sparse_temperature.svg | Sensorpunkte ohne Annotationen |
| dense_temperature_windows.svg / sparse_temperature_windows.svg | Standardauswahl, normalerweise 3/6/9 mit Rahmen |
| dense_local_windows.svg / sparse_local_windows.svg | Herausgezogene lokale Windows der Standardauswahl |
| dense_all_local_windows.svg / sparse_all_local_windows.svg | Alle zwölf lokalen Windows getrennt in einem 3 × 4 Raster |
| dense_controller.svg / sparse_controller.svg | Gemeinsamer MAT mit zwölf Agent-Tokens |
| dense_actuators_12.svg / sparse_actuators_12.svg | Zwölf Aktuatoren in der jeweiligen Farbfamilie |
| two_plume_temperature_field.svg | Vollständiger gespeicherter 96 × 64 Temperaturzustand |
| temperature_legend.svg | Separate Temperaturskala |
| agent_palettes.svg | Agentfarben beider Seiten mit Agentnummern |
| distillation_arrow.svg | Distillationspfeil |

Alle SVGs verwenden transparente Hintergründe, normale Vektorelemente und
editierbaren Text (Arial, Helvetica/sans-serif als Fallback).
Die Galerie stellt Dense und Sparse direkt gegenüber und lässt sich nach
Agentenzahl und Stil filtern. Die PNGs sind ausschließlich Vorschauen.

## Erzeugen

Vom Repository-Root aus:

~~~powershell
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl"
~~~

Die SVG-Erzeugung benötigt JLD2, JSON und Julia-Standardbibliotheken.
Sie startet keine Simulation, kein Training und keine neue Kandidatenauswahl.
Beim bloßen include wird nichts erzeugt. Die zehn Vergleichsauswahlen werden
bei jedem normalen Aufruf hergestellt.

~~~powershell
# Nur Quellen und globale/lokale Maskenzuordnung prüfen:
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl" --check-only

# Eigene Standardauswahl (ein bis zwölf verschiedene Agenten):
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl" --agents 2,4,8,10 --output-dir "Revision/Main Figure/custom"

# Alle zwölf Agenten auch als Standardansicht:
julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl" --agents all --output-dir "Revision/Main Figure/all_agents"
~~~

Weitere Optionen: --experiment-id ID für eine andere eingefrorene Package-8-
GO-GC-Auswahl und --state-file PATH für einen kompatiblen gespeicherten
Oceananigans-Two-Plume-Checkpoint. Das Standardexperiment ist fest
260830_231109 und wechselt nicht automatisch zu neueren Ergebnissen.

Optional erzeugt der folgende Aufruf mit Node.js und sharp alle Einzel-PNGs
unter assets/preview/ sowie die drei Vergleichsübersichten:

~~~powershell
node "Revision/Main Figure/render_preview.cjs"
~~~

Ein abweichender Asset-Ordner kann als erstes Argument übergeben werden.
Die SVG-Erzeugung benötigt Node und sharp nicht.

## Datenherkunft und wissenschaftliche Bedeutung

- Zustand: RBmodel300.jld2, der gespeicherte Two-Plume-Ausgangszustand aus
  Revision/Run_Files/FixedIC_MAT.jl. Er dient hier zur Illustration der Methode
  mit einer Varying-IC-Maske, nicht als bestimmter Varying-IC-Testrollout.
- Maske: Revision/Package8/results/260830_231109/go-gc/analysis/selected_test_candidate.jld2.
- Kandidat: 2e7a2411c43d08d0c2041d86, Run p8_260830_231109_go_gc_s_0p02_r01,
  Update 99850, Regularisierungsstärke 0.02, Maskenschwelle 0.003.
- Bereits vor dem Test eingefrorene Validation-Auswahl;
  Validation-MSE 0.00030600613603989284.
- Aktiv: **1/32 GC-Gruppen, 12/384 Sensororte, 36/1152 skalare Messwerte**.
  Aktiv sind die horizontalen Sensorindizes 1,5,9,...,45 in der vertikalen
  Sensorzeile 4 (von unten), jeweils gemeinsam T/w/u.

Die physischen Proben werden exakt an den Feldindizes x=1:2:95 und z=1:8:57
ausgelesen, mit denselben Interior-Slices wie im Run-File. Es wird nicht
interpoliert. Die gemeinsame Sensorposition wird durch das Temperatur-
Zellzentrum x=(ix-0.5)·2π/96, z=(iz-0.5)·2/64 dargestellt; der physikalische
Seitenquotient bleibt erhalten. Die lokalen Ausschnitte zeigen ein Indexraster.

Die Temperaturpalette und ihre festen Grenzen **[1, 2.5]** entsprechen der
ersten Iteration und dem bisherigen RBC-Sensorplot. Inaktive Punkte erhalten
exakt **#F2F2F2**. Die aktiven Punkte haben auf beiden Seiten und in allen
Varianten exakt dieselben Temperaturwerte und Farben.

Die Darstellung zeigt den physischen Zustand **vor dem sinusförmigen
positional encoding**. Der echte MAT-Temperaturinput enthält diesen bekannten
Positionsanteil zusätzlich. Die vollständige Dreikanal-Maske wird weiterhin
gegen die gespeicherte lokale Eingangsmaske aller zwölf Agenten geprüft.

Die Agentzentren sind die horizontalen Sensorindizes 3,7,...,47. Jedes Window
enthält modulo 48 genau sieben Spalten links und rechts vom Zentrum und alle
acht Höhen: 120 Orte / 360 skalare Eingänge. Die Maske behält pro Window
vier Orte / zwölf skalare Eingänge. Der MAT verarbeitet die zwölf
Agentbeobachtungen gemeinsam; die Symbole stellen keine zwölf unabhängig
parametrisierten Netze dar.

provenance.json enthält Quell- und Generator-Hashes, Maskenidentität,
Sensorindizes, Agentfarben, alle Vergleichsauswahlen und die Zuordnungen
aller zwölf Windows. Die endgültige Figure-Komposition und der PDF-Export
erfolgen weiterhin manuell.
