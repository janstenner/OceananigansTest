# Paket 7 und 8 — vollständiger Regularisierer- und Gruppierungsvergleich

Stand: 2026-08-30

## Ziel und Abgrenzung

Paket 7 untersucht Fixed IC, Paket 8 anschließend Varying IC. Beide Pakete
vergleichen alle vorgesehenen Apprentice-Regularisierer und Gruppierungen über
drei gepaarte Seeds, Offline-Expert-Matching und `active_inputs`.

Die Produktionsruns sind ein vollständig neuer Experimentblock mit einem neuen
Master-Seed. **GO-SC und GR-SC werden nicht aus Paket 6 übernommen**, sondern
ebenso wie alle anderen Kombinationen neu trainiert. Paket 6 dient nur als
methodische Vorstudie; seine Runs und Seeds gehen nicht in Paket 7/8 ein.

Closed-loop-Auswahl und finale Simulation werden erst nach Abschluss der
Trainings- und Paretoanalyse festgelegt. Testdaten beeinflussen weder Strength-,
Checkpoint-, Threshold- noch Maskenauswahl.

## Datenprotokolle

### Paket 7: Fixed IC

Training, Offline-Validation und Closed-loop-Auswertung verwenden das gemeinsame
Fixed-IC-Szenario. Es gibt keinen unabhängigen Testsplit; die finale Auswertung
ist daher keine Held-out-Generalisation.

### Paket 8: Varying IC

- Training-Corpus ausschließlich für Apprentice-Training.
- Validation-Corpus für Offline-Paretoanalyse und spätere Closed-loop-Auswahl.
- Test-Corpus ausschließlich für die terminale Evaluation bereits eingefrorener
  Modelle, Masken und Thresholds.

Varying-Rollouts speichern Split, Basis-Seed, Spiegelung, Offset,
Evaluationsseed, Episode, Kontrollschritt und Simulationszeit.

## Trainingsprotokoll

- neuer P7-Master-Seed `20_260_850`;
- daraus drei Apprentice- und Batch-Reihenfolge-Seedpaare;
- dieselben drei Seedpaare für alle Methoden, Gruppierungen und Strength-Versionen;
- 70.000 Updates für Fixed, 50.000 für Varying;
- Regressions-Lernrate `2e-4`;
- Trainingsbatchgröße 50/100 und Validation-Batchgröße 200/512 für
  Fixed/Varying;
- Validation ab Update 0 alle 25 Updates;
- kein ergebnisabhängiger Stopp außer bei technischem numerischem Versagen;
- kein Fine-Tuning nach Maskierung oder Training;
- mindestens eine aktive Trainingsgruppe.

Die P7-Seedpaare (Apprentice/Batch-Reihenfolge) sind
`r01=996898248/1207818757`, `r02=1452413696/1103313457` und
`r03=497948374/1844296950`.

## Methodenmatrix und Strength-Versionen

Pro IC-Protokoll müssen alle acht Kombinationen ausführbar sein:

| Methode | Channel-Coupled (GC) | Separate-Channel (SC) |
|---|---:|---:|
| GO | ja | ja |
| GR | ja | ja |
| Group Lasso | ja | ja |
| Standard-GrOWL | ja | ja |

Jede Kombination startet in Paket 7 standardmäßig drei Strengths mit je drei
Replicates, also neun Trainingsworker und genau einen gemeinsamen Analyzer.
Explizite `--strength`-Argumente können dieses editierbare Raster pro Launch
ersetzen. Methode, Gruppierung, Strength, Protokoll und Seed sind Bestandteil
der Run-Identität. Für die finale Berichterstattung wird je Kombination nur die
anhand der Validation-Ergebnisse passendste Strength verwendet.

Als erste Startwerte werden die Werte aus `GrOWL/MAT_expert_apprentice.jl`
übernommen:

| Methode | Fixed | Varying | alter Name |
|---|---:|---:|---|
| GO | `0.09` | `0.025` | `gro_asc` |
| GR | `0.00004` | `0.0001` | `weighted` |
| Group Lasso | `0.0001` | `0.00012` | `lasso` |
| Standard-GrOWL | `0.00006` | `0.0004` | `growl` |

Diese Werte sind Startpunkte und keine bereits ausgewählten finalen Strengths.

Die aktuelle P7-Matrix verwendet jeweils den Faktor 2,5:

| Kombination | Strengths |
|---|---|
| GO-GC | `(0.008, 0.02, 0.05)` |
| GO-SC | `(0.016, 0.04, 0.1)` |
| GR-GC | `(0.000024, 0.00006, 0.00015)` |
| GR-SC | `(0.000048, 0.00012, 0.0003)` |
| Group-Lasso-GC | `(0.000032, 0.00008, 0.0002)` |
| Group-Lasso-SC | `(0.000064, 0.00016, 0.0004)` |
| GrOWL-GC | `(0.000048, 0.00012, 0.0003)` |
| GrOWL-SC | `(0.000096, 0.00024, 0.0006)` |

Diese Raster sind ausschließlich in `Package7Study.jl` definiert und dort
direkt manuell editierbar. Der Analyzer erhält beim Start die tatsächlich
verwendeten Strengths und erwartet kein festes Raster.

## Threshold- und Evaluationsprotokoll

Jeder Basischeckpoint wird alle 25 Updates mit genau vier Mask-Thresholds
ausgewertet:

```text
(0.0, 0.003, 0.006, 0.012)
```

Für jeden Threshold werden eine gruppenkonsistente Maske, `active_inputs` und
das Validation-Expert-Action-Matching bestimmt. Threshold `0.0` bezeichnet die
native exakte Nullmaske und wird immer validiert. Ein Hard-Threshold wird nur
dann validiert und als Evaluationszeile gespeichert, wenn er gegenüber dem
nativen Kandidaten desselben Checkpoints die Anzahl aktiver Gruppen reduziert.
Identische erfolgreiche Masken dürfen nur einmal berechnet werden; ihre
threshold-spezifischen Evaluationszeilen bleiben erhalten.

Die Thresholds sind absolut. Zunächst wird für jeden Apprentice-Input die
L1-Summe seiner Embedding-Gewichte berechnet. Die Importance einer Gruppe ist
das Maximum dieser Input-Importances innerhalb der Gruppe. Eine Gruppe bleibt
aktiv, wenn ihre Importance strikt größer als der Threshold ist. Würde eine
Maske alle Gruppen entfernen, wird deterministisch die stärkste Gruppe
reaktiviert; bei Gleichstand gewinnt der kleinste Gruppenindex. `active_inputs`
zählt die global expandierten eindeutigen Sensor-Kanal-Inputs. Diese
Threshold-Importance ist von der unveränderten L2-Importance des GR-Trainings
getrennt.

Die Pareto-Dominanz verwendet ausschließlich:

1. weniger oder gleich viele `active_inputs`;
2. kleineres oder gleiches Validation-Expert-Matching;

Wie in P6 Fixed gilt `validation_matching <= 0.01` als Qualitätskriterium.
Die gepoolte Pareto-CSV kennzeichnet dies pro Punkt, und der Plot zeigt die
Grenze als gestrichelte horizontale Linie.
3. mindestens eine strikte Verbesserung.

Der Threshold selbst ist keine dritte Zielgröße.

Wenn mindestens einer der vier Punkte zur aktuellen Run-Pareto-Front beiträgt,
wird das Apprentice-Modell dieses Updates einmal gespeichert. Für die
beitragenden Punkte werden Threshold, Maske, Importance-Metadaten und Messwerte
als Kandidaten erhalten. Trägt kein Punkt bei, bleiben nur die vier schlanken
Evaluationszeilen; die Masken und das Modell werden verworfen. Wird ein früherer
Kandidat später vollständig dominiert, darf sein Modell nach verifizierter
Garbage Collection entfernt werden, während seine Evaluationszeilen erhalten
bleiben.

Nach dem Training werden Fronten pro Run und eine über alle neun Runs gepoolte
Front pro Konfiguration gebildet. Ein Analyseworker wartet auf alle übergebenen
Strength-Replicate-Kombinationen und erzeugt einen Pareto-Plot mit sämtlichen
Evaluationspunkten, Threshold-Farben, Seed-Markern und hervorgehobener gepoolter
Front. Danach friert er validation-only den Kandidaten mit den wenigsten
`active_inputs` unter `validation_matching <= 0.01` ein und führt mit dessen
gespeicherter Maske eine 200-Schritte-Fixed-Testepisode aus. Actions,
Reward-Verlauf und direkter `state_Nu`-Verlauf werden vollständig gespeichert.
Modelle mehrerer Threshold-Kandidaten desselben Updates werden nicht dupliziert.

## Paket-7-Ausführung

Die Implementierung liegt in `Revision/Package7`. Ohne Launcher-Filter werden
alle acht Konfigurationen mit ihren drei Strengths gestartet. `--config` startet
eine einzelne Kombination; wiederholtes `--strength` ersetzt für diesen Launch
deren Standardraster. Eine Kombination besteht standardmäßig aus neun
Trainingsworkern und einem Analyse-/Plot-Waiter. Jeder Launch schreibt in einen
kurzen timestamp-basierten Ergebnisordner; der Analyzer erhält dessen ID und die
tatsächlich gestarteten Strengths explizit. Die finale Strength-Auswahl erfolgt
erst nach Sichtung der Validation-Ergebnisse.

## Spätere umfassende Closed-loop-Phase

Die einzelne sparsity-orientierte Testepisode jedes Analyseworkers ersetzt
nicht die spätere methodenübergreifende Auswahlphase. Für diese werden Anzahl
und weitere Auswahlregeln separat festgelegt. Vorgesehen sind mindestens ein
Matching-orientierter, ein Knee- und ein sparsity-orientierter Kandidat.
Gemessen werden Reward, vollständiges `state_Nu()`, Stabilität, Fehlläufe und
Laufzeit. Nicht alle Offline-Kandidaten werden simuliert.

## Noch offen für die spätere Auswahlphase

1. Validation-Regel zur finalen Strength-Auswahl je Kombination;
2. Closed-loop-Budget und Kandidatenauswahl nach Abschluss des Trainings;
3. eigener Master-Seed und finale Detailkonfiguration für Paket 8.

Alle Konfigurationen, Seeds, Evaluationspunkte, Kandidaten und final berichteten
Werte bleiben maschinenlesbar zu Run, Checkpoint, Threshold und Maske
rückverfolgbar.
