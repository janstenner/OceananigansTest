# Paket 4 — MAT versus IPPO

Stand: 2026-07-31

## Implementierungsstand

Die Pipeline liegt unter `Revision/MAT_IPPO_Comparison`: persistenter Seed- und
IC-Plan, atomare MAT/IPPO-Worker, validierender Paket-3-Importer, begrenzter
persistenter tmux-Slot-Pool und inkrementeller Collector. Null-Episoden-
Initialisierungen und vollständige deterministische Rollouts sind für alle vier
Protokoll-/Algorithmus-Kombinationen geprüft. Das Paket bleibt bis zur
Durchführung und Auswertung der Produktionsruns offen.

Dieses Dokument legt die vollständige technische und experimentelle Planung
für Paket 4 fest.

## Ziel

Verglichen werden parameter-sharing IPPO und die `modified_full`-Variante von
MAT unter Fixed IC und Varying IC.

Die Zielgröße bei vollständiger Durchführung ist:

| Protokoll | MAT | IPPO | Episoden pro Run |
|---|---:|---:|---:|
| Fixed IC | 10 | 10 | 2.000 |
| Varying IC | 10 | 10 | 4.000 |

Die tatsächliche Zahl vorhandener Runs darf während der Durchführung kleiner
oder später auch größer sein. Weder Launcher noch Collector dürfen zehn Runs
als technische Voraussetzung fest eincodieren.

Fünf MAT-Runs je Protokoll sollen aus der Paket-3-Stabilitätsstudie übernommen
werden. Damit sind bei vollständiger Wiederverwendung noch 30 Trainingsjobs
neu auszuführen:

- 5 Fixed-IC-MAT-Runs,
- 10 Fixed-IC-IPPO-Runs,
- 5 Varying-IC-MAT-Runs,
- 10 Varying-IC-IPPO-Runs.

## Konfigurationen

MAT verwendet ausschließlich `modified_full`:

- `useSeparateValueChain = true`,
- `useLayerNorm = false`,
- `useSelfAttentionFirst = false`,
- `use_mus = true`,
- Dropout `0.0`.

IPPO verwendet die zu MAT größenangepassten Fixed-IC- beziehungsweise
Varying-IC-Konfigurationen aus `Revision/Run_Files`.

Beide Algorithmen behalten alle übrigen protokollspezifischen Einstellungen
aus den Package-2-Run-Files. In neuen Dateinamen, Metadaten und Abbildungen
wird die Bezeichnung `IPPO` und nicht das ältere Kurzlabel `PPO` verwendet.

## Dynamischer Seedplan ohne feste Run-Nummern

Run-Nummern wie 1–5 oder 6–10 sind nur eine informelle Beschreibung der zwei
geplanten Durchführungsschritte. Sie werden nicht als Voraussetzung in
Dateinamen, Planung oder Collector festgeschrieben.

Ein persistenter Paket-4-Plan enthält nur tatsächlich erzeugte oder
importierte Seedpaare. Jeder Eintrag besitzt eine unveränderliche, aus den
Seeds beziehungsweise einer stabilen ID abgeleitete Run-ID und mindestens:

- `run_id`,
- `run_seed`,
- `ic_seed`,
- eine Batch-ID des erzeugenden oder importierenden Aufrufs,
- Entstehungsart `generated` oder `imported_package3`,
- Erstellungszeit,
- für Varying IC die vorab festgelegte IC-Folge,
- aus den kanonischen Ergebnis- und Validation-Dateien ableitbarer Status der
  vier möglichen Jobs Fixed-MAT, Fixed-IPPO, Varying-MAT und Varying-IPPO.

Der Plan wird vor dem Start von Workern atomar geschrieben. Worker dürfen
keine Seeds erzeugen oder auswählen. MAT und IPPO desselben Planeintrags
erhalten ihre Seeds explizit aus diesem Plan. Es gibt daher keinen „ersten“
Worker, der den Seed für den zweiten festlegt, und keine Race Condition bei
parallel gestarteten Algorithmuspaaren.

Neu generierte Seeds müssen gegen alle bereits im Paket-4-Plan vorhandenen
Seeds auf Kollisionen geprüft werden. Importierte Paket-3-Seeds werden erst
dann in den Plan aufgenommen, wenn die entsprechenden Ergebnisdateien
tatsächlich vorhanden und validiert sind. Beim ersten Start dürfen daher keine
zukünftigen Paket-3-Seeds oder Platzhalter für sie angelegt werden.

## Launcher-Semantik

Der Launcher erhält mindestens folgende Optionen:

```text
--n_runs N
--look_for_imports true|false
--max-workers N
--protocol all|fixed|varying
--preview
--overwrite
```

`--n_runs` bezeichnet die Zahl der Seedpaare, die der jeweilige Aufruf neu
vorbereiten soll. Daraus können je nach Importstatus und Protokollauswahl
unterschiedlich viele Trainingsjobs entstehen.

### Neue Seedpaare

Der erste geplante Start lautet beispielsweise:

```bash
launch_tmux.sh --n_runs 5 --look_for_imports false
```

Dieser Aufruf:

1. lädt den bestehenden Paket-4-Plan oder erzeugt einen leeren Plan,
2. erzeugt genau fünf neue Seedpaare,
3. erzeugt für jeden Varying-IC-Run einmalig die gemeinsame IC-Folge,
4. schreibt den erweiterten Plan atomar,
5. plant für jedes Seedpaar Fixed-MAT, Fixed-IPPO, Varying-MAT und
   Varying-IPPO,
6. startet die fehlenden Jobs unter Beachtung von `--max-workers`.

Bei Auswahl beider Protokolle entstehen aus fünf neuen Seedpaaren genau 20
Jobs. Mit dem Default `--max-workers 20` können diese gleichzeitig starten.

### Importierte Paket-3-Seedpaare

Nach Abschluss der Paket-3-Runs lautet der zweite geplante Start:

```bash
launch_tmux.sh --n_runs 5 --look_for_imports true
```

Dieser Aufruf verwendet zunächst dieselbe validierte Importlogik wie das
eigenständig ausführbare Importer-Skript. Sie sucht
nach noch nicht importierten, vollständigen `modified_full`-Ergebnissen aus
Paket 3 und übernimmt bis zu der mit `--n_runs` angeforderten Zahl von
Seedpaaren. Es werden keine Ersatz-Seeds generiert, wenn zu wenige
importierbare Ergebnisse vorliegen. Stattdessen beendet sich die Planung vor
dem Workerstart mit einer klaren Meldung, damit nicht unbemerkt ein anderes
Experimentdesign entsteht.

Nach erfolgreichem Import erkennt der Launcher MAT als bereits vollständig
und startet für diese Seedpaare nur die fehlenden IPPO-Jobs. Bei fünf
importierten Seedpaaren und beiden Protokollen sind dies zehn neue Jobs.

Wiederholte Aufrufe müssen idempotent sein: vorhandene vollständige Ergebnisse
werden übersprungen, aktive Jobs werden nicht doppelt gestartet und fehlende
oder zuvor fehlgeschlagene Jobs können erneut eingeplant werden.

## Parallelisierung

Der Default für `--max-workers` ist 20. Ein Worker bearbeitet genau einen Job,
also genau eine Kombination aus Seedpaar, Protokoll und Algorithmus.

Wenn MAT und IPPO für dasselbe Seedpaar beide fehlen, werden sie als Paar
benachbart eingeplant und bei ausreichender Kapazität in zwei getrennten tmux-
Slots gestartet. Sie trainieren parallel, nicht sequenziell in einem
gemeinsamen Julia-Prozess. Liegen mehr Jobs als Slots vor, arbeitet jeder Slot
seine Queue nacheinander ab.

Alle tmux-Slots sind detached, schreiben getrennte Logs, überleben die SSH-
Trennung und bleiben auch nach Abschluss ihrer Queue als interaktive Shell
offen. Die Parallelitätsgrenze muss vom Benutzer kleiner gesetzt werden
können, wenn RAM oder CPU des Servers nicht für 20 gleichzeitige Prozesse
reichen.

## Varying-IC-Paarung

Gleiche `ic_seed`-Werte allein reichen nicht als Paarungsnachweis. Für jeden
Varying-IC-Planeintrag wird vor dem Workerstart eine Folge von exakt 4.000
Auswahlen gespeichert. Jeder Eintrag enthält:

- Split `:train`,
- `base_seed`,
- `mirror`,
- `offset`.

MAT und IPPO erhalten dieselbe Folge explizit über die bereits vorhandenen
Parameter von `generate_random_init`. Interne Zufallsziehungen eines
Algorithmus können die IC-Folge dadurch nicht verschieben.

Bei importierten Paket-3-Ergebnissen muss die geplante Folge vollständig mit
deren gespeichertem `initial_condition_trace` übereinstimmen. Eine Abweichung
blockiert den Import beziehungsweise den gepaarten Vergleich.

Fixed IC verwendet für beide Algorithmen denselben unveränderten
Initialzustand. Gleiche `run_seed`-Werte bedeuten bei den unterschiedlichen
Architekturen keine gleichen Gewichte, stellen aber die reproduzierbare
Seedpaarung her.

## Paket-3-Importer

Das Importer-Skript ist ein eigener, auch manuell ausführbarer Julia-Einstieg.
Der Paket-4-Launcher ruft es bei `--look_for_imports true` vor jedem
Workerstart auf.

Es akzeptiert nur Paket-3-Dateien mit:

- Status `complete`,
- Konfiguration `modified_full`,
- Protokoll `fixed` oder `varying`,
- exakt 2.000 beziehungsweise 4.000 abgeschlossenen Episoden,
- vollständigem Reward-Verlauf,
- vorhandenen `run_seed`- und `ic_seed`-Werten,
- bei Varying IC einem vollständigen IC-Trace,
- Seeds ohne Konflikt mit einem anderen Paket-4-Planeintrag.

Für ein importiertes Seedpaar müssen die Fixed-IC- und Varying-IC-Dateien aus
Paket 3 dieselben Seedwerte tragen. Standardmäßig wird nur ein vollständiges
Paar beider Protokolle übernommen; unvollständige Paare werden gemeldet und
noch nicht als einer der angeforderten Imports gezählt.

Die verifizierten Paket-3-JLD2-Dateien werden atomar in die kanonische
Paket-4-Ergebnisstruktur kopiert. Der Importer überschreibt keine abweichende
Zieldatei. In der Kopie beziehungsweise zugehörigen Importmetadaten werden
mindestens Quellpfad, SHA-256-Hash der Quelle, Importzeit und Paket-4-Run-ID
festgehalten.

Der Collector behandelt importierte und neu trainierte MAT-Ergebnisse danach
gleich. Die finalen Agenten aus Paket 3 sind ausreichende End-Checkpoints und
werden nicht nachträglich durch Trainingsepisoden-basierte Checkpoints ersetzt.

## Training und Speicherung

Jeder neue Worker:

1. lädt genau eines der vier Package-2-Run-Files,
2. setzt die explizit übergebenen Seeds,
3. initialisiert Environment, Agent und Hook,
4. prüft die erwartete Konfiguration und Parameterzahl,
5. verwendet bei Varying IC die gespeicherte IC-Folge,
6. trainiert exakt 2.000 beziehungsweise 4.000 Episoden,
7. speichert unmittelbar danach atomar eine JLD2-Datei.

Jede vollständige Ergebnisdatei enthält mindestens:

- Algorithmus und Protokoll,
- Paket-4-Run-ID,
- `run_seed` und `ic_seed`,
- Trainingsbudget und Zahl abgeschlossener Episoden,
- Rewards und Fehlläufe,
- Laufzeit,
- finalen Agenten,
- relevante Hyperparameter und Parameterzahl,
- Varying-IC-Trace,
- Start- und Endzeit, Host und Julia-Version,
- Git-Stände und SHA-256-Hashes der wesentlichen Quelldateien.

Fehler werden mit Stacktrace und Teilverlauf atomar gespeichert. Locks
verhindern doppelte Jobs für dasselbe Ergebnisziel.

## Checkpoint- und Expert-Auswahl

Es werden keine umfangreichen Zwischencheckpointfolgen und keine Auswahl nach
der besten Trainingsepisode verwendet. Für jeden Run ist der Agent nach dem
festen Trainingsbudget der Kandidat.

Alle finalen MAT-Checkpoints werden anschließend auf derselben vorab
festgelegten Validation-Suite evaluiert. Der MAT-End-Checkpoint mit der besten
Validation-Performance wird getrennt für Fixed IC und Varying IC als Expert
für die nachfolgende Expert-Apprentice-Distillation ausgewählt.

Diese Regel gilt identisch für importierte Paket-3-Runs und neu trainierte
Runs. Sie ist im Paper kompakt beschreibbar: festes Trainingsbudget, Auswahl
des finalen Checkpoints über ein gemeinsames Validation-Set, kein Zugriff auf
das Test-Set während der Auswahl.

## Validation-Protokoll

Die Evaluation verwendet deterministische Policy-Actions, also die
vorhergesagten Mittelwerte ohne zusätzliches Samplingrauschen.

### Fixed IC

Jeder finale Checkpoint wird in genau einem vollständigen Rollout auf demselben
Fixed-IC-Zustand bewertet. Der Episodenreturn ist der Validation-Score.

### Varying IC

Verwendet wird der einzige Basiszustand aus dem Validation-Split mit dem
kartesischen Produkt aus:

- `mirror ∈ {false, true}`,
- `offset ∈ {0, 20}`.

Dies ergibt die vier festen Fälle:

```text
(false, 0)
(false, 20)
(true, 0)
(true, 20)
```

Der arithmetische Mittelwert der vier Episodenreturns ist der
Validation-Score. Die vier Einzelwerte werden ebenfalls gespeichert. Der
Test-Split wird erst nach abgeschlossener Expert-Auswahl verwendet.

IPPO-End-Checkpoints werden auf denselben Validation-Fällen evaluiert, damit
die finale Algorithmusauswertung vergleichbar bleibt; die Expert-Auswahl für
die Distillation erfolgt ausschließlich unter den MAT-Kandidaten.

## Collector und variable Run-Anzahl

Der Collector durchsucht die Paket-4-Ergebnisstruktur und sammelt alle
vorhandenen vollständigen, gültigen Dateien ein. Er setzt weder zehn Runs noch
eine bestimmte Nummerierung voraus.

- Fehlende Protokolle, Algorithmen, Seedpaare oder Einzeljobs sind zulässig.
- Importierte und neue MAT-Dateien werden gemeinsam ausgewertet.
- Vorhandene fehlerhafte Dateien werden gemeldet und übersprungen.
- Paarungsprüfungen werden überall durchgeführt, wo MAT und IPPO desselben
  Planeintrags vorhanden sind.
- Plots und Tabellen nennen stets die tatsächlich verwendete Zahl `n`.
- Alte Plots für inzwischen nicht vorhandene Daten werden vor einer erneuten
  Sammlung entfernt.

Der Collector kann nach jedem Teilbatch erneut ausgeführt werden und erzeugt
jeweils den aktuellen Ergebnisstand.

## Lernkurvenstatistik

Wie in den bisherigen `gather_results.jl` wird zuerst für jeden einzelnen Run
ein Rolling Mean über 50 Episoden berechnet. Erst danach werden Runs
aggregiert.

Die primäre Darstellung enthält:

- sehr transparente Linien aller Einzelruns,
- den Median als kräftige Linie,
- das 25%- bis 75%-Quantil als asymmetrisches Band,
- den Mittelwert als dünne gestrichelte Linie.

Der Median passt natürlich zum Quantilband und bleibt robust gegenüber
einzelnen Fehlläufen. Der zusätzlich gezeigte Mittelwert bildet trotzdem den
durchschnittlichen Einfluss schlechter oder außergewöhnlich guter Seeds ab.

Zusätzlich werden pro Episode Mittelwert, Standardabweichung, Median, Q25,
Q75 und die bedingten mittleren Abweichungen oberhalb und unterhalb des
Mittelwerts maschinenlesbar gespeichert.

## Zu erzeugende Auswertungen

Der Collector erzeugt grundsätzlich alle folgenden Artefakte. Welche davon in
das Paper übernommen werden, wird später entschieden.

1. Fixed-IC-Lernkurven für MAT und IPPO mit Einzelruns, Median, IQR und
   Mittelwert.
2. Varying-IC-Lernkurven in derselben Darstellung.
3. Gepaarte Differenzkurven `MAT − IPPO` mit Nulllinie, Median und IQR.
4. Finale Trainingsperformance pro Run als Mittelwert der letzten 100
   Episoden, mit Rohpunkten und gepaarten Verbindungslinien.
5. Validation-Performance aller finalen MAT- und IPPO-Checkpoints.
6. Getrennte Expert-Rankings der finalen MAT-Checkpoints für Fixed IC und
   Varying IC.
7. Laufzeitvergleich mit Rohpunkten, Median und IQR.
8. Zahl und Anteil fehlerhafter Episoden pro Run.
9. Diagnostisches Ranking nach bestem Rolling-Window-Trainingswert, ohne dass
   dieses Ranking zur Checkpoint- oder Expert-Auswahl verwendet wird.
10. Einzelrun-Lernkurven und eine Übersicht auffälliger oder fehlgeschlagener
    Runs.
11. CSV- und JLD2-Zusammenfassungen aller Rohwerte und Statistiken.
12. Maschinenlesbare Paarungstabelle mit Run-ID, Seeds, Quelldateien,
    Importstatus und Validierungsfällen.

Statische PNG- beziehungsweise PDF-Abbildungen werden aus denselben
gespeicherten Zusammenfassungen erzeugt; interaktive HTML-Plots kommen hinzu,
wenn dies mit dem installierten Plot-Backend ohne eine zweite Auswertungslogik
praktikabel ist.

## Kompakte Implementierungsstruktur

Die Implementierung soll aufgeräumt, möglichst direkt und kompakt bleiben.
Insbesondere darf die Ablaufsteuerung nicht in viele sehr kleine
Unterfunktionen oder schwer nachvollziehbare Funktionsketten aufgespalten
werden.

Vorgesehen sind nur wenige klar abgegrenzte Einstiegspunkte:

- ein gemeinsames Julia-Experimentmodul für Plan, Workerlogik und Speicherung,
- ein dünnes Worker-CLI,
- ein eigenständiges Importer-Skript,
- ein tmux-Launcher,
- ein Collector inklusive Validation und Plotting.

Fixed/Varying und MAT/IPPO sollen nicht durch vier weitgehend duplizierte neue
Experimentimplementierungen abgebildet werden. Stattdessen werden die vier
Package-2-Run-Files direkt eingebunden und durch wenige gut sichtbare
Fallunterscheidungen konfiguriert.

Hilfsfunktionen sind nur dann vorgesehen, wenn sie eine eigenständige,
wiederverwendete Aufgabe kapseln, etwa atomisches Speichern oder das Lesen von
JLD2-Metadaten. Einmalige Ablaufschritte bleiben möglichst in der
aufrufenden Funktion, damit Seedplanung, Import, Jobauswahl, Training und
Sammlung ohne Funktions-Spaghetti nachvollziehbar bleiben.

## Abnahmekriterien

Paket 4 ist technisch umgesetzt, wenn:

- Seedpläne schrittweise und ohne Platzhalter erweitert werden können,
- neue MAT- und IPPO-Paare tatsächlich parallel mit identischen Planseeds
  starten,
- der Importer Paket-3-`modified_full`-Runs sicher übernimmt,
- Start und Neustart ohne manuelle Run-Nummern möglich sind,
- die Varying-IC-Folgen innerhalb jedes Algorithmuspaars identisch sind,
- Fixed- und Varying-Worker ihre exakten Episodenbudgets einhalten,
- der Collector jede vorhandene Teilmenge ohne feste Run-Anzahl verarbeitet,
- Validation und Expert-Auswahl ausschließlich finale Checkpoints verwenden,
- alle vorgesehenen Rohdaten, Statistiken und Plotartefakte reproduzierbar
  erzeugt werden.

Der experimentelle Abschluss von Paket 4 ist erreicht, wenn die angestrebten
zehn MAT- und zehn IPPO-Runs je Protokoll vorliegen, alle finalen Checkpoints
validiert wurden und je ein Fixed-IC- und Varying-IC-MAT-Expert ohne Zugriff
auf den Test-Split ausgewählt wurde.
