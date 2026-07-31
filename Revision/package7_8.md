# Paket 7 und 8 — Apprentice-Distillation, Pareto-Auswahl und Evaluation

Stand: 2026-07-31

## Ziel und Abgrenzung

Paket 7 und 8 führen die in Paket 6 aufgebaute Pareto-Infrastruktur in die vollständigen Fixed-IC- und Varying-IC-Experimente des Papers über.
Die in Paket 6 erzeugten GO- und GR-Runs werden wiederverwendet und nicht ohne Grund erneut ausgeführt.

Paket 6 beantwortet die methodische Frage nach Sensitivität und Seed-Stabilität von GO.
Paket 7 und 8 beantworten die anwendungsbezogene Frage, welche Apprentice-Modelle und Masken nach Offline-Auswahl, Closed-loop-Validation und gegebenenfalls finaler Testevaluation berichtet werden.

Die wesentlichen Erweiterungen gegenüber Paket 6 sind:

- die noch fehlenden Regularisierer und Gruppierungsvarianten
- grouped Lasso und Standard-GrOWL
- Hard-Threshold-Kandidaten
- Closed-loop-Validation weniger Pareto-Kandidaten
- finale Evaluation der ausgewählten Modelle
- vollständige Paper-Tabellen, Sensorplots und Performancevergleiche

## 1. Gemeinsame Voraussetzungen

Vor Beginn werden festgeschrieben:

- die final verwendeten Fixed-IC- und Varying-IC-Experts
- die MAT-Apprentice-Architektur
- die Teacher-Rollout-Datensätze
- die Apprentice-Trainingsbudgets
- die Checkpoint- und Validationintervalle
- die betrachteten Methoden und Gruppierungsvarianten
- die Definition des Expert-Matching-Fehlers
- die Definition der Gesamtzahl aktiver Inputs
- die Threshold-Importance und der vollständige Thresholdbereich
- das maximale Budget für Closed-loop-Validation und Testevaluation

Alle Randomisierungen werden über gespeicherte Seeds kontrolliert.
Die finale Testevaluation darf keine spätere Modell-, Masken- oder Thresholdentscheidung beeinflussen.

## 2. Methoden- und Gruppierungsmatrix

Die vollständige Matrix enthält, soweit die jeweilige Methode nach Paket 6 im Paper verbleibt:

- GO mit channel-coupled grouping
- GO mit separate-channel grouping
- GR mit channel-coupled grouping
- GR mit separate-channel grouping
- grouped Lasso mit channel-coupled grouping
- grouped Lasso mit separate-channel grouping
- Standard-GrOWL mit channel-coupled grouping
- Standard-GrOWL mit separate-channel grouping

Die aus Paket 6 übernommenen GO-Stärken werden nicht erneut anhand von Testresultaten angepasst.
Falls Paket 6 eine einzelne GO-Stärke auswählt, wird nur diese weitergeführt.
Falls eine konservative und eine aggressive Stärke komplementäre Pareto-Bereiche abdecken, dürfen höchstens diese beiden weitergeführt werden.

Grouped Lasso bezeichnet die gleiche overlap-konsistente Gruppenstruktur wie bei den anderen Methoden mit einer einheitlichen Regularisierungsgewichtung.
Standard-GrOWL verwendet die absteigende Gewichtsanordnung und bleibt klar von GO mit aufsteigenden Gewichten getrennt.

## 3. Datenprotokoll für Paket 7: Fixed IC

Für Fixed IC sind Training-, Validation- und Testdaten gemäß der getroffenen Festlegung identisch.
Es existiert daher kein unabhängiger held-out Testsplit.

Das Fixed-IC-Protokoll umfasst:

- Teacher-Rollouts aus dem festgelegten Fixed-IC-Expert
- Apprentice-Training auf diesen Rollouts
- Offline-Expert-Matching auf denselben Daten
- Pareto-Auswahl anhand von Expert Matching und Anzahl aktiver Inputs
- Closed-loop-Auswertung weniger ausgewählter Kandidaten auf dem Fixed-IC-RBC-Problem
- abschließende ausführliche Closed-loop-Berichterstattung des ausgewählten Modells

Die abschließende Fixed-IC-Auswertung wird nicht als unabhängige Generalisierungsevaluation bezeichnet.
Sie dokumentiert die kontrollierte Performance im untersuchten Fixed-IC-Szenario.

## 4. Datenprotokoll für Paket 8: Varying IC

Für Varying IC werden die drei Corpus-Splits strikt getrennt verwendet:

- Trainingbasen ausschließlich für Expert-Rollouts und Apprentice-Training
- Validationbasis ausschließlich für Offline-Pareto-Messungen und Closed-loop-Validation
- Testbasen ausschließlich für die einmalige finale Evaluation der ausgewählten Modelle

Jeder Rollout speichert mindestens:

- Corpus-Split
- Basis-Seed
- Spiegelung
- horizontalen Offset
- Run-Seed
- Episode
- Kontrollschritt
- Simulationszeit

Die Testbasen werden weder für Hyperparameterwahl noch für Checkpoint-, Masken-, Threshold- oder GO-Power-Auswahl verwendet.

## 5. Wiederverwendung der Runs aus Paket 6

Alle kompatiblen Paket-6-Kandidaten werden mit ihren vollständigen Metadaten in die Kandidatenmengen von Paket 7 und 8 übernommen.

Insbesondere werden wiederverwendet:

- GO mit channel-coupled grouping unter Fixed IC
- GO mit channel-coupled grouping unter Varying IC
- die nativen GR-Referenzkandidaten mit channel-coupled grouping unter Fixed IC
- die nativen GR-Referenzkandidaten mit channel-coupled grouping unter Varying IC
- die in Paket 6 erzeugten Run-, Parameter- und gepoolten Pareto-Metadaten

Ein Paket-6-Run wird nur erneut ausgeführt, wenn:

- seine Konfiguration nicht mit der finalen Produktionskonfiguration übereinstimmt
- erforderliche Metadaten fehlen
- für eine spätere Threshold-Expansion benötigte Checkpoints nicht gespeichert wurden
- ein technischer Fehler die Wiederverwendung verhindert

GO-Kandidaten aus Paket 6 sind ohne zusätzliche Threshold-Expansion direkt wiederverwendbar.
Für GR muss geprüft werden, ob die in Paket 7 und 8 gewünschte kleine Threshold-Expansion aus den gespeicherten Checkpoints erzeugt werden kann.
Falls Paket 6 dafür nicht genügend Basischeckpoints erhalten hat, werden ausschließlich die betroffenen GR-Runs erneut mit vollständiger Kandidatenerzeugung ausgeführt.

## 6. Apprentice-Training und Checkpointauswertung

Jeder neue Apprentice-Run verwendet:

- ein festes Maximalbudget
- regelmäßig festgelegte Checkpoints
- ein festes Offline-Validationintervall
- einen gespeicherten Initialisierungsseed
- einen gespeicherten Datenreihenfolgeseed
- identische Teacher-Daten innerhalb gepaarter Vergleiche

Manuelles Stoppen anhand eines gerade akzeptabel erscheinenden Kompromisses aus Expert Matching und Sparsity entfällt.
Ergebnisabhängige Stopps bleiben auf technische Sicherheitsfälle wie NaN, Inf oder einen eindeutigen numerischen Zusammenbruch begrenzt.

Jeder ausgewertete Basischeckpoint speichert mindestens:

- Methode
- IC-Protokoll
- Gruppierungsvariante
- Regularisierungskonfiguration
- Apprentice-Seed
- Trainingsschritt
- Checkpoint-ID
- Validation-Expert-Matching
- native Anzahl aktiver Inputs
- native Maske
- Pfad oder Referenz auf das Apprentice-Modell

## 7. Methodenspezifische Kandidatenerzeugung

### GO

GO erzeugt seine primäre Kandidatenmenge mit Threshold $\tau=0$.
Die durch GO selbst exakt auf null gesetzten Gruppen bestimmen die native Maske.

Die Paket-6-Kandidaten werden übernommen.
Neue GO-Runs werden nur für fehlende Gruppierungsvarianten oder die nach Paket 6 ausdrücklich weitergeführten GO-Stärken ausgeführt.

### GR

GR erhält zunächst eine native Kandidatenmenge mit $\tau=0$.
Da GR in den bisherigen Experimenten in geringem Umfang von nachträglichem Thresholding betroffen war, kann zusätzlich derselbe vorab definierte Thresholdprozess angewendet werden.

Die native und die threshold-assisted GR-Front werden getrennt identifizierbar gehalten.

### Grouped Lasso

Grouped Lasso wird zunächst nativ mit $\tau=0$ ausgewertet.
Es wird erwartet, dass bei Messung ausschließlich exakt null gesetzter Gruppen nur geringe native Sparsity entsteht.

Anschließend wird für jeden ausgewerteten Basischeckpoint der vollständige vorab definierte Thresholdbereich angewendet.

### Standard-GrOWL

Standard-GrOWL wird zunächst nativ mit $\tau=0$ ausgewertet.
Es wird erwartet, dass ohne Hard Thresholding alle oder nahezu alle Inputs aktiv bleiben.

Da dann alle nativen Checkpoints dieselbe volle Inputzahl besitzen, wäre auf der nativen Front pro Run nur der Checkpoint mit dem besten Validation-Expert-Matching nichtdominiert.
Deshalb muss die Threshold-Expansion vor der Pareto-Reduktion der GrOWL-Checkpoints erfolgen.

## 8. Hard-Threshold-Kandidaten

Der Threshold ist ein Parameter der Kandidatenerzeugung und keine dritte Pareto-Zielgröße.

Ein Kandidat wird identifiziert durch

$$
c =
(\text{method},
\text{seed},
\text{checkpoint},
\tau,
\text{mask}),
$$

während die Pareto-Dominanz ausschließlich anhand von

$$
N_{\mathrm{active}}(c)
$$

und

$$
L_{\mathrm{match,val}}(c)
$$

bestimmt wird.

Der Thresholdbereich beginnt bei $\tau=0$ und wird in vorab festgelegten Stufen, beispielsweise $0.005$, bis zu einem vor Produktionsbeginn festgelegten Maximalwert durchlaufen.
Der Bereich darf nach Einsicht in Validation- oder Testresultate nicht willkürlich erweitert werden.

Vor Beginn muss festgelegt werden, ob der Threshold auf eine absolute oder normalisierte Group Importance angewendet wird.
Diese Definition bleibt anschließend über Methoden, Seeds und IC-Protokolle konsistent und wird mit jedem Kandidaten gespeichert.

Für jeden Basischeckpoint gilt die Reihenfolge:

1. Group Importance berechnen.
2. Alle vorab festgelegten Thresholds anwenden.
3. Channel-coupled Gruppen immer gemeinsam maskieren.
4. Identische Masken über verschiedene Thresholds deduplizieren.
5. Jede unterschiedliche Maske auf dem Validation Set auswerten.
6. Erst danach Pareto-Dominanz prüfen.

Es findet kein Fine-Tuning nach dem Thresholding statt.

Mehrere Masken desselben Basischeckpoints referenzieren dasselbe Apprentice-Modell.
Der Threshold, die Maske und ihre Messwerte werden getrennt gespeichert.

## 9. Warum die Reihenfolge bei Lasso und GrOWL entscheidend ist

Ein nativ dominierter Lasso- oder GrOWL-Checkpoint kann nach Thresholding einen nichtdominierten sparse Kandidaten erzeugen.
Deshalb dürfen Basischeckpoints dieser Methoden nicht vor der Threshold-Expansion allein anhand ihrer nativen Punkte verworfen werden.

Für Standard-GrOWL gilt typischerweise:

$$
N_{\mathrm{active}}^{(t,0)} = N_{\mathrm{full}}
$$

für alle Trainingsschritte $t$.
Die nichttriviale Performance-Sparsity-Punktwolke entsteht erst aus den Kandidaten mit $\tau>0$.

Ein Apprentice-Modell wird pro Checkpoint höchstens einmal gespeichert.
Trägt nach vollständiger Threshold-Expansion keine seiner Masken zu einer relevanten Pareto-Front bei, kann das vollständige Modell verworfen werden.

## 10. Pareto-Dominanz und Archive

Ein Kandidat $A$ dominiert einen Kandidaten $B$, wenn

$$
N_{\mathrm{active}}^A \leq N_{\mathrm{active}}^B
$$

und

$$
L_{\mathrm{match,val}}^A \leq L_{\mathrm{match,val}}^B,
$$

wobei mindestens eine Ungleichung strikt sein muss.

Folgende Archive werden gebildet:

- Front pro Run
- Front pro Methode und Gruppierungsvariante
- Front pro Regularisierungskonfiguration
- gepoolte Front über Apprentice-Seeds
- gemeinsame Front aller Methoden für Fixed IC
- gemeinsame Front aller Methoden für Varying IC

Alle Punkte behalten ihre Erzeugungsmetadaten.
Bei identischer Maske wird der Kandidat mit dem besten Validation-Expert-Matching bevorzugt.
Bei identischer Anzahl aktiver Inputs bleibt für die geometrische Front der Kandidat mit dem kleinsten Validation-Expert-Matching.

## 11. Native und threshold-assisted Ergebnisansichten

### Native Sparsification

Die native Ansicht verwendet für alle Methoden ausschließlich $\tau=0$.
Sie beantwortet:

> Welche Methoden erzeugen bereits während des Trainings exakt sparse Modelle?

Diese Darstellung macht sichtbar, ob GO und GR native Nullgruppen erzeugen und ob grouped Lasso oder Standard-GrOWL ohne Maskenextraktion dicht bleiben.

### Threshold-assisted deployment

Die threshold-assisted Ansicht enthält alle vorab erzeugten Threshold-Kandidaten.
Sie beantwortet:

> Welchen besten deploybaren Performance-Sparsity-Trade-off erreicht jede Methode einschließlich dokumentierter Maskenextraktion?

Die gemeinsame faire Methodenauswahl erfolgt anhand der zwei Ergebnisgrößen Anzahl aktiver Inputs und Validation-Expert-Matching.
Der Threshold bleibt als erklärender Parameter erhalten.

## 12. Darstellungen der Offline-Kandidaten

Für Fixed IC und Varying IC werden getrennte Darstellungen erzeugt.

Die zentrale 2D-Darstellung verwendet:

- x-Achse: Anzahl aktiver Inputs
- y-Achse: Validation-Expert-Matching
- Farbe oder Symbol: Methode und Gruppierungsvariante
- Hervorhebung der nichtdominierten Punkte

Für grouped Lasso, Standard-GrOWL und gegebenenfalls threshold-assisted GR wird zusätzlich eine 3D-Punktwolke erzeugt:

- x-Achse: Anzahl aktiver Inputs
- y-Achse: Validation-Expert-Matching
- z-Achse: Threshold
- Hervorhebung der anhand von x und y nichtdominierten Punkte

Es findet keine Dominanzbewertung anhand der Thresholdhöhe statt.
Ein kleinerer Threshold ist nicht automatisch besser als ein größerer.

Zusätzlich werden Run- und Seedzugehörigkeit, Checkpointschritt und Maskengröße maschinenlesbar erhalten.

## 13. Hypothese für Lasso und GrOWL

Die vorab festgehaltene Erwartung lautet:

> Grouped Lasso und Standard-GrOWL erzeugen ohne Hard Thresholding wenig oder keine exakte Sparsity. Auch nach einem fairen Threshold-Sweep wird erwartet, dass ihr erreichbarer Expert-Matching-Sparsity-Trade-off ungünstiger als bei GO und GR ausfällt.

Diese Aussage ist eine Hypothese und keine Auswahlregel.
Lasso- oder GrOWL-Kandidaten dürfen zur gemeinsamen Pareto-Front beitragen, wenn die Ergebnisse der Hypothese widersprechen.

Der Threshold-Sweep verhindert, dass Lasso oder GrOWL allein wegen fehlender mathematisch exakter Nullwerte unfair als vollständig dicht bewertet werden.

## 14. Auswahl für Closed-loop-Validation

Die Offline-Pareto-Front dient als günstiger Filter.
Nur eine kleine vorab budgetierte Auswahl wird in der RBC-Simulation ausgeführt.

Die Auswahl soll unterschiedliche Bereiche abdecken:

- einen konservativen Kandidaten mit sehr gutem Expert Matching
- einen Kandidaten im Knee-Bereich
- einen aggressiven Kandidaten mit wenigen aktiven Inputs

Die Auswahl kann methodenspezifische Repräsentanten enthalten, wenn dies für einen fairen Vergleich erforderlich ist.
Identische oder nahezu identische Masken werden vor der Simulation dedupliziert.

Closed-loop-Validation bedeutet:

- der Apprentice steuert die Simulation selbst
- seine Aktionen beeinflussen alle folgenden Zustände
- gemessen werden kumulativer Reward, globaler Nusselt-Verlauf, Stabilität, Fehlläufe und Laufzeit
- die Resultate dürfen zur finalen Modellauswahl verwendet werden

Die Zahl der simulierten Kandidaten und Episoden wird vor Beginn begrenzt und anschließend berichtet.

## 15. Finale Auswahl und Testevaluation

Die finale Auswahl erfolgt ausschließlich aus Offline-Validation und Closed-loop-Validation.
Der Threshold wird gemeinsam mit Modell und Maske eingefroren.

### Paket 7: Fixed IC

Das ausgewählte Fixed-IC-Modell erhält eine ausführliche finale Closed-loop-Auswertung auf dem festgelegten Fixed-IC-Szenario.
Da kein unabhängiger Split existiert, ist dies keine held-out Generalisierungsevaluation.

### Paket 8: Varying IC

Nach Abschluss aller Entscheidungen wird das ausgewählte Varying-IC-Modell einmalig auf den Testbasen evaluiert.
Die Testevaluation ist closed loop und verwendet vorab festgelegte Basiszustände, Spiegelungen, Offsets und Evaluationsseeds.

Nach Einsicht in diese Testergebnisse werden weder Modell noch Maske, Threshold, GO-Stärke oder Checkpoint geändert.

## 16. Metadaten und Rückverfolgbarkeit

Jeder Run und Kandidat speichert mindestens:

- Paket- und Experimentversion
- Git-Commit
- Julia-Version und Manifestidentifikation
- Fixed- oder Varying-IC-Protokoll
- Methode
- Gruppierungsvariante
- Regularisierungskonfiguration
- GO-Power beziehungsweise GR-Konfiguration
- Apprentice-Seed
- Netzinitialisierungsseed
- Datenreihenfolgeseed
- Trainingsschritt
- Checkpoint-ID
- Threshold
- Definition und Normalisierung der Group Importance
- Masken-Hash und vollständige Maske
- Anzahl aktiver Inputs
- Validation-Expert-Matching
- Pareto-Status und zugehörige Front
- Pfad oder Referenz zum Apprentice-Modell

Für Varying-IC-Rollouts und Simulationen kommen hinzu:

- Corpus-Split
- Basis-Seed
- Spiegelung
- Offset
- Episode
- Kontrollschritt
- Simulationszeit
- Evaluationsseed

Für Closed-loop-Läufe werden außerdem kumulativer Reward, Nusselt-Verlauf, Laufstatus, Abbruchgrund und Laufzeit gespeichert.

## 17. Ergebnisartefakte

Paket 7 erzeugt für Fixed IC:

- vollständige Offline-Pareto-Daten
- native und threshold-assisted Methodenvergleiche
- 2D- und 3D-Pareto-Darstellungen
- Closed-loop-Vergleiche der ausgewählten Kandidaten
- finale Expert-Apprentice-Trajektorien
- Sparsity-Tabelle
- Sensor-Pattern-Plots
- Rohdaten und aggregierte Statistiken

Paket 8 erzeugt analog für Varying IC:

- vollständige Offline-Pareto-Daten auf dem Validation-Split
- native und threshold-assisted Methodenvergleiche
- 2D- und 3D-Pareto-Darstellungen
- Closed-loop-Validation der ausgewählten Kandidaten
- finale Closed-loop-Testevaluation
- Return-Verteilungen und repräsentative Trajektorien
- Sparsity-Tabelle
- Sensor-Pattern-Plots
- Rohdaten und aggregierte Statistiken

Alle berichteten Zahlen bleiben zu Run, Seed, Checkpoint, Maske und Threshold zurückverfolgbar.

## 18. Festgelegter Cut

Paket 7 und 8 enthalten:

- Wiederverwendung kompatibler Kandidaten aus Paket 6
- fehlende GO- und GR-Gruppierungsvarianten
- grouped Lasso
- Standard-GrOWL
- methodenspezifische Threshold-Kandidatenerzeugung
- Offline-Pareto-Archive
- begrenzte Closed-loop-Validation
- finale Fixed-IC-Berichterstattung
- finale Varying-IC-Testevaluation

Nicht enthalten sind:

- nachträgliche Änderung des Thresholdbereichs anhand von Testresultaten
- Fine-Tuning nach Hard Thresholding
- Simulation aller Offline-Kandidaten
- Nutzung der Varying-IC-Testbasen zur Auswahl
- neue Reward-Estimator- oder Reward-Modul-Experimente
- neue physikalische Regime oder Layouts

## Vor Produktionsbeginn festzulegen

Vor den ersten Paket-7- und Paket-8-Produktionsruns werden noch numerisch festgelegt:

- Checkpoint- und Validationintervall
- vollständiger Thresholdbereich und Maximalwert
- absolute oder normalisierte Group Importance
- genaue Definition der global eindeutigen aktiven Inputs
- konservative, mittlere und aggressive Inputbereiche
- Zahl der Closed-loop-Kandidaten pro Methode oder gemeinsamer Front
- Closed-loop-Episodenbudget
- Fixed-IC-Berichtsprotokoll
- Varying-IC-Validation- und Testfälle
