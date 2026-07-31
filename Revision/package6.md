# Paket 6 — Minimale GO-Sensitivitätsstudie auf RBC-Daten

Stand: 2026-07-31

## Ziel

Paket 6 ist eine kleine, vollständig offline durchführbare Sensitivitätsstudie auf echten RBC-Teacher-Rollouts.
Es ist weder eine Toy-Studie noch eine zweite vollständige Apprentice-Ergebnisstudie.

Die zentrale Frage lautet:

> Erzeugt GO bei moderater Änderung seiner Regularisierungsstärke wiederholt einen brauchbaren empirischen Pareto-Bereich zwischen Expert Matching und Anzahl aktiver Inputs, und ist dieser erreichbare Bereich über Apprentice-Seeds hinreichend stabil?

Ein GO- oder GR-Run wird dabei nicht durch einen einzelnen Endcheckpoint repräsentiert.
Jeder Run liefert über seinen Trainingsverlauf eine Menge von Kandidaten

$$
\mathcal S
=
\left\{
\left(
N_{\mathrm{active}}^{(t)},
L_{\mathrm{match,val}}^{(t)}
\right)
\right\}_{t\in\mathcal T},
$$

wobei $N_{\mathrm{active}}^{(t)}$ die Anzahl aktiver Inputs und $L_{\mathrm{match,val}}^{(t)}$ das Expert Matching auf dem Validation Set am ausgewerteten Trainingsschritt $t$ bezeichnet.

## 1. Motivation für ein Pareto-Archiv

GO und GR drücken Expert-Matching-Fehler und Gewichte nicht monoton auf einen praktisch konvergierten Endzustand.
Während des Trainings können Expert Matching, Anzahl aktiver Inputs oder beide Größen sprunghaft schlechter und später wieder besser werden.
Ein Run kann sich daher mehrfach an einen guten Performance-Sparsity-Trade-off annähern und wieder davon entfernen.

Der aktuelle Trainingspunkt darf deshalb nicht mit dem Ergebnis des gesamten Runs gleichgesetzt werden.
Stattdessen führt jeder Run ein externes Pareto-Archiv über alle regelmäßig ausgewerteten Checkpoints.

Das Archiv verschlechtert sich im Zeitverlauf nicht.
Ein neuer Kandidat erweitert oder verbessert die bisherige empirische Front, oder er wird von einem früheren Kandidaten dominiert.

Das Pareto-Verfahren ist eine nachgelagerte Archivierungs- und Auswahlstrategie.
Es wird nicht behauptet, dass GO oder GR selbst Multi-Objective-Pareto-Optimierer sind.

## 2. Voraussetzungen einfrieren

Paket 6 beginnt erst, nachdem die MAT-Stabilitätsstudie und die Expertwahl abgeschlossen sind.
Vor Beginn werden folgende Bestandteile festgeschrieben:

- die verwendete MAT-Apprentice-Architektur
- der Fixed-IC-Expert
- der Varying-IC-Expert
- die Observation- und Action-Normalisierung
- die Trainingsdatenformate
- die Gruppenkonstruktion
- der maximale Apprentice-Trainingsumfang
- das Auswertungsintervall für Pareto-Kandidaten

Dadurch untersucht Paket 6 tatsächlich GO und nicht gleichzeitig wechselnde Architekturen oder Experts.
Die für Paket 6 erzeugten Teacher-Daten werden anschließend unverändert in Paket 7 und 8 weiterverwendet.

## 3. Variierter Parameter

Variiert wird ausschließlich die multiplikative GO-Stärke `growl_power_used`.
Die geordnete Gewichtsfolge

$$
\lambda_i = \frac{i-1}{n}
$$

bleibt unverändert.
Ihre Form, die Proximalfrequenz und alle anderen Optimierungsparameter bleiben konstant.

Im bestehenden Code sind die bisherigen GO-Zentralwerte:

- Fixed IC: `0.09`
- Varying IC: `0.025`

Vorgesehen ist folgender vorab festgelegter Dreipunktsweep:

| Protokoll | niedrig | bisheriger Wert | hoch |
|---|---:|---:|---:|
| Fixed IC | 0.045 | 0.09 | 0.18 |
| Varying IC | 0.0125 | 0.025 | 0.05 |

Die drei Werte entsprechen jeweils $0.5\times$, $1\times$ und $2\times$ des bisherigen Werts.
Nach Beginn der Produktionsruns wird der Sweep nicht anhand der Resultate erweitert oder verschoben.

Das im Code als `power` bezeichnete Argument ist keine mathematische Potenz.
Es ist der Skalierungsfaktor der GO-Gewichte.

## 4. Experimentumfang

Untersucht werden beide Datenprotokolle:

- Fixed IC
- Varying IC

Beide Fälle werden einbezogen, weil der bestehende Code verschiedene GO-Stärken für sie verwendet.
Eine reine Fixed-IC-Studie würde den Varying-IC-Wert nicht absichern.

Der experimentelle Sensitivitätssweep verwendet ausschließlich channel-coupled grouping.
Dies ist die praktisch wichtigste Variante, bei der ein physischer Sensorort gemeinsam über seine Komponenten behandelt wird.
Separate-channel grouping wird technisch geprüft, erhält jedoch keinen eigenen vollständigen Sensitivitätssweep.

## 5. Seeds und Paarung

Jede GO-Stärke wird mit drei Apprentice-Seeds untersucht.
Innerhalb eines Seeds bleiben über alle GO-Stärken identisch:

- Netzinitialisierung
- Reihenfolge der Trainingsdaten
- Teacher-Daten
- Trainingsbudget
- Optimizer
- Gruppierung
- Batchaufteilung

Damit wird innerhalb eines Seeds ausschließlich die GO-Stärke verändert.
Der GO-Umfang beträgt

$$
2\ \text{Protokolle}
\times
3\ \text{GO-Stärken}
\times
3\ \text{Seeds}
=
18\ \text{GO-Runs}.
$$

## 6. GR als Referenz

GR erhält keinen eigenen Sensitivitätssweep.
Es wird mit seiner bisherigen nominalen Konfiguration als Referenz ausgeführt:

- Fixed IC: `0.00004`
- Varying IC: `0.0001`
- jeweils drei mit den GO-Runs gepaarte Seeds
- dieselben Daten und Trainingsbudgets

Damit kommen sechs GR-Runs hinzu.
Paket 6 umfasst insgesamt 24 offline durchgeführte Apprentice-Runs.

Jeder GR-Run erzeugt dasselbe Pareto-Archiv wie ein GO-Run.
Der GR-Vergleich soll keine allgemeine Überlegenheit eines Verfahrens nachweisen.
Er dient der Beantwortung folgender Fragen:

- Ist die Streuung der erreichbaren GO-Fronten auffällig größer als bei GR?
- Ist das Maskenverhalten von GO bei vergleichbaren Inputbudgets weniger reproduzierbar?
- Liefert GO einen vergleichbaren oder erkennbar anderen erreichbaren Performance-Sparsity-Bereich?

Lasso und Standard-GrOWL gehören nicht in Paket 6.
Sie werden später gemeinsam mit den eigentlichen Baselines behandelt.

## 7. Correctness-Audit vor den Produktionsruns

Vor den Produktionsruns müssen kleine deterministische Tests bestätigen:

1. Jede Inputzeile gehört genau zu der erwarteten Gruppe.
2. Keine Gruppen überlappen sich unbeabsichtigt.
3. Channel-coupled Gruppen enthalten alle vorgesehenen Komponenten eines Sensororts.
4. Die Gruppennormen werden korrekt sortiert.
5. Die GO-Gewichte werden den sortierten Normen korrekt zugeordnet.
6. Die proximal veränderten Normen werden den ursprünglichen Gruppen korrekt zurückgegeben.
7. Exakt auf null gesetzte Gruppen erzeugen die erwartete Maske.
8. Alle Zufallsoperationen verwenden den Run-RNG.

Besonders zu behandeln ist die bisherige zufällige Wiederherstellung von Nullgruppen in `apply_growl`.
Sie darf in der Sensitivitätsstudie keine unkontrollierte zusätzliche Zufälligkeit einführen.

Bevorzugt wird die zufällige Wiederherstellung aus der Studie entfernt und ein fast vollständig geprunter Lauf als degeneriertes Ergebnis festgehalten.
Alternativ muss die Wiederherstellung deterministisch erfolgen.

## 8. Training ohne manuelles Stoppen

Alle 24 Runs erhalten dasselbe feste Maximalbudget innerhalb ihres jeweiligen IC-Protokolls.
Ergebnisabhängige Stopps werden für die Studie deaktiviert.
Dies betrifft insbesondere:

- Stoppen anhand der gesamten Weight Sum
- Stoppen aufgrund manuell motivierter Loss-Verläufe
- Stoppen anhand eines gewünschten Sparsity-Niveaus

Ein technischer Sicherheitsstopp bleibt ausschließlich für folgende Fälle zulässig:

- `NaN` oder `Inf`
- ungültige Modellparameter
- eindeutiger numerischer Zusammenbruch

Ein stärker regularisierter Lauf darf nicht früher beendet werden, nur weil er schneller sparse wird.

## 9. Kandidatenerzeugung und Pareto-Archiv pro Run

Jeder Run wird in einem vorab festgelegten festen Intervall auf dem zugehörigen Validation Set ausgewertet.
Das Intervall muss klein genug sein, um die Sprünge zwischen gutem Expert Matching und hoher Sparsity zuverlässig abzubilden.

Jeder Kandidat erhält mindestens folgende Metadaten:

- Methode
- GO-Stärke beziehungsweise GR-Konfiguration
- IC-Protokoll
- Apprentice-Seed
- Trainingsschritt
- Validation-Expert-Matching
- Anzahl aktiver Inputs
- binäre Maske
- numerischer Status

Ein Kandidat $A$ dominiert einen Kandidaten $B$, wenn

$$
N_{\mathrm{active}}^A \leq N_{\mathrm{active}}^B
$$

und

$$
L_{\mathrm{match,val}}^A \leq L_{\mathrm{match,val}}^B,
$$

wobei mindestens eine der beiden Ungleichungen strikt sein muss.

Für die Archivierung gilt:

- Ein dominierter neuer Kandidat wird vollständig in den Messwerten protokolliert, sein vollständiges Apprentice-Modell muss jedoch nicht gespeichert werden.
- Ein nichtdominierter neuer Kandidat wird mit Modell und Maske in das Pareto-Archiv aufgenommen.
- Dominiert ein neuer Kandidat ältere Pareto-Punkte, können deren vollständige Modelle aus dem aktiven Archiv entfernt werden.
- Kleine Metadaten und Masken dürfen für die spätere Dynamik- und Stabilitätsanalyse erhalten bleiben.
- Bei identischer Anzahl aktiver Inputs reicht für die geometrische Front der Kandidat mit dem kleinsten Validation-Expert-Matching.

Ein innerhalb seines eigenen Runs dominierter Kandidat kann auch auf einer später gepoolten Front nicht mehr benötigt werden, solange sein Dominator erhalten bleibt.

## 10. Kein nachträgliches Hard Thresholding

Paket 6 zählt ausschließlich die durch GO beziehungsweise GR selbst exakt auf null gesetzten Gruppen.
Es wird kein nachträglicher Near-zero-Threshold verwendet.

Dadurch werden zwei getrennte Effekte nicht vermischt:

1. Sensitivität des Regularisierungstrainings
2. Sensitivität der nachträglichen Maskenextraktion

Hard Thresholding wird erst in Paket 7 und 8 als eigener Kandidatenschritt in die Pareto-Erzeugung aufgenommen.

## 11. Drei Ebenen der Pareto-Auswertung

### A. Pareto-Front pro Run

Für jede Kombination aus IC-Protokoll, Methode, Regularisierungsstärke und Seed wird eine eigene Front gebildet.
Sie beschreibt den während eines einzelnen Trainingslaufs erreichbaren Trade-off.

### B. Pareto-Front pro Regularisierungsstärke

Die drei Seed-Fronten einer GO-Stärke werden zusammengeführt.
Dabei bleibt sichtbar:

- welche Teile der gepoolten Front von welchem Seed stammen
- ob ein guter Bereich nur durch einen einzelnen glücklichen Seed erreicht wurde
- wie stark sich die Seed-Fronten überlappen
- welchen Sparsity-Bereich die jeweilige GO-Stärke innerhalb des festen Budgets erreicht

Für GR wird dieselbe gepoolte Referenzfront erzeugt.

### C. Gemeinsame Front über alle GO-Stärken

Die Kandidaten aller drei GO-Stärken werden anschließend zusammengeführt.
Jeder Punkt behält seine Kennzeichnung für GO-Stärke, Seed und Trainingsschritt.

Diese Front zeigt:

- welche GO-Stärke zu welchem Bereich beiträgt
- ob eine Stärke die anderen weitgehend dominiert
- ob verschiedene Stärken komplementäre konservative und aggressive Bereiche abdecken

Die gemeinsame Front dient in Paket 6 der Sensitivitätsanalyse.
Sie bestimmt noch nicht das finale Paper-Modell.

## 12. Messgrößen

### Primäre Messgrößen

Für jeden ausgewerteten Kandidaten werden zwei Hauptgrößen berechnet:

1. **Expert Matching auf dem Validation Set:** mittlere quadratische Abweichung zwischen den vorhergesagten Aktionsmitteln von Expert und Apprentice.
2. **Gesamtzahl aktiver Inputs:** Anzahl der nach Gruppierung und Maskenbildung tatsächlich beibehaltenen Inputs.

Für Fixed IC sind Training-, Validation- und Testdaten entsprechend dem festgelegten Protokoll identisch.
Das Expert Matching ist dort keine Generalisierungsmessung, sondern eine kontrollierte Optimierungs- und Sensitivitätsmessung.

Für Varying IC wird das Expert Matching auf dem separaten Validation-Split berechnet.
Die Testbasen werden in Paket 6 nicht verwendet.

### Sekundäre Messgrößen

Zusätzlich werden gespeichert:

- Trainingsloss über den Trainingsverlauf
- Validation-Expert-Matching über den Trainingsverlauf
- aktive Inputs über den Trainingsverlauf
- Gruppennormen
- finale und archivierte binäre Masken
- Jaccard-Ähnlichkeit der Masken zwischen Seeds bei vergleichbaren Inputbudgets
- Auswahlhäufigkeit jedes Sensororts beziehungsweise Kanals
- Erreichbarkeitsrate vorab definierter Inputbudgets
- Anzahl und Art numerischer Fehlläufe
- Laufzeit

## 13. Auswertung

### A. Trajektorien und Run-Fronten

Für ausgewählte Runs werden alle Checkpointpunkte und die jeweils nichtdominierten Punkte gemeinsam gezeigt.
Dadurch wird sichtbar, wie der aktuelle Trainingspunkt wiederholt an die empirische Front heranläuft und von ihr wegspringt.

### B. Sensitivitäts- und Performance-Sparsity-Darstellung

Es wird je ein Panel für Fixed IC und Varying IC erstellt.
Die gemeinsame Darstellung verwendet:

- x-Achse: Anzahl aktiver Inputs
- y-Achse: Validation-Expert-Matching
- Farbe: GO-Stärke
- einzelne Punkte oder Linien: Run-Fronten der Apprentice-Seeds
- gesonderte Referenzdarstellung für GR

Zusätzlich wird pro GO-Stärke berichtet, welche Inputbereiche innerhalb des festen Budgets erreicht wurden.

### C. Vergleich bei festen Inputbudgets

Vor der Auswertung werden konservative, mittlere und aggressive Inputbereiche festgelegt.
Für jeden Seed wird der beste Frontpunkt im jeweiligen Bereich bestimmt.

Pro Bereich werden berichtet:

- bestes Validation-Expert-Matching pro Seed
- Median oder Mittelwert mit Streuung
- tatsächlich erreichte Anzahl aktiver Inputs
- Anteil der Runs, die den Bereich überhaupt erreicht haben

Ein nicht erreichter Bereich wird als Ergebnis ausgewiesen und nicht durch Extrapolation ersetzt.

### D. Maskenstabilität

Masken werden nur bei vergleichbaren Inputbudgets miteinander verglichen.
Eine Tabelle oder Matrix berichtet:

- mittlere paarweise Jaccard-Ähnlichkeit
- minimale und maximale Jaccard-Ähnlichkeit
- Auswahlhäufigkeit der Sensororte
- getrennte Werte nach GO-Stärke, Inputbereich und IC-Protokoll

Dadurch wird vermieden, die geringe Jaccard-Ähnlichkeit zweier Masken primär durch stark unterschiedliche Maskengrößen zu erklären.

## 14. Entscheidungsregel

GO bleibt eine reguläre Hauptmethode, wenn:

- der Correctness-Audit bestanden ist
- die nominalen Runs ohne numerische Fehlschläge enden
- die GO-Stärke den erreichbaren Sparsity-Bereich nachvollziehbar beeinflusst
- mindestens ein Bereich mit brauchbarem Expert Matching und deutlicher Inputreduktion wiederholt über Seeds erreicht wird
- gute Frontpunkte nicht ausschließlich aus einem einzelnen glücklichen Seed stammen
- die Streuung und Maskenstabilität gegenüber GR nicht fundamental schlechter sind

Ein schlechter Endcheckpoint allein ist kein Ausschlussgrund, wenn der Run zuvor reproduzierbar brauchbare nichtdominierte Kandidaten erzeugt hat.

GO wird sekundär oder explorativ behandelt, wenn:

- Gruppensortierung oder Rückzuordnung nicht zuverlässig funktionieren
- identische Konfigurationen stark erratische Fronten liefern
- die Regularisierungsstärke keinen nachvollziehbaren Einfluss auf den erreichbaren Sparsity-Bereich besitzt
- brauchbares Expert Matching nur bei praktisch dichter Maske möglich ist
- gute Frontpunkte nur aus einzelnen zufälligen Ausnahmeruns stammen
- das Ergebnis maßgeblich von zufälliger Nullgruppenwiederherstellung abhängt

Die numerische Akzeptanzgrenze für brauchbares Expert Matching und die Inputbereiche werden vor Beginn der Produktionsruns festgelegt.

Falls eine GO-Stärke die anderen über Seeds hinweg weitgehend dominiert, wird diese Konfiguration in Paket 7 und 8 übernommen.
Falls verschiedene Stärken komplementäre Frontbereiche abdecken, werden höchstens eine konservative und eine aggressive Stärke weitergeführt.

## 15. Festgelegter Cut

Paket 6 umfasst:

- zwei RBC-Datenprotokolle
- channel-coupled grouping
- drei GO-Stärken
- drei Apprentice-Seeds
- eine nominale GR-Referenz
- Pareto-Archive über regelmäßig ausgewertete Trainingscheckpoints
- ausschließlich Offline-Auswertung
- keine Testdaten
- kein Hard Thresholding
- keine Closed-Loop-Simulation pro Sensitivitätspunkt
- keine Toy-Probleme
- keine Lasso- oder Standard-GrOWL-Sweeps

Der Umfang bleibt bei 18 GO-Runs und 6 GR-Runs.
Die teuren Closed-Loop-Auswertungen folgen erst in Paket 7 und 8 für wenige Pareto-Kandidaten.

## Noch festzulegende Begriffsentscheidung

Empfohlen wird, die Gesamtzahl aktiver Inputs als Anzahl der global eindeutigen, beibehaltenen Sensor-Kanal-Paare nach Auflösung der überlappenden Fenster zu definieren.
Diese Größe ist für die spätere Deploymentaussage aussagekräftiger als die Anzahl mehrfach vorkommender lokaler Inputzeilen.
