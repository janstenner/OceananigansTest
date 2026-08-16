# Paket 6 — SC-GO-Sensitivitäts- und Stabilitätsstudie

Stand: 2026-08-10

## Ziel und Abgrenzung

Paket 6 ist eine eigenständige wissenschaftliche Studie auf den echten
RBC-Teacher-Rollouts. Es untersucht ausschließlich Separate-Channel-Gruppierung
(`group_channels=false`) und native Regularisierer-Sparsity. Die technische
Strength-Kalibrierung war interne Vorarbeit zur Festlegung eines übersichtlichen,
Pareto-relevanten Strength-Grids; sie wird im Paper nicht erwähnt und gehört
nicht zu den Produktionsresultaten.

Ein Run wird nicht durch seinen letzten Checkpoint repräsentiert. GO und GR
können sich wiederholt an einen guten Sparsity-/Matching-Bereich annähern, aus
ihm herausspringen und sich wieder erholen. Deshalb bleiben sämtliche
Validation-Evaluationen erhalten und werden als Trainingsverlauf, wissenschaftliche
Front sowie Stabilitätsprozess ausgewertet.

Paket 7 und 8 ergänzen später Channel-Coupled- und Separate-Channel-Varianten,
Hard Thresholding und weitere Regularisierer. Paket 6 wählt kein finales
Paper-Modell aus.

## Eingefrorenes Trainingsprotokoll

Für Fixed und Varying gelten dieselben fünf GO-Stärken:

```text
0.0015, 0.003, 0.006, 0.01, 0.03
```

GR wird als nominale Offline-Referenz geführt:

- Fixed IC: `0.00004`
- Varying IC: `0.0001`

Weitere Festlegungen:

- drei neue gepaarte Replicates;
- deterministischer Seedplan aus `StableRNG(20260810)`;
- der Kalibrierungsseed `600601` wird nicht wiederverwendet;
- innerhalb eines Replicates teilen alle fünf GO-Runs und der GR-Run die
  Apprentice-Initialisierung und Batchreihenfolge;
- Regression-Lernrate `2e-4`;
- 35.000 Fixed- beziehungsweise 50.000 Varying-Updates;
- Trainingsbatchgröße 50 für die vollständige Fixed-Wiederholung und weiterhin
  100 für Varying; Validation-Batchgröße 200 (der vollständige gemeinsame
  Fixed-Corpus) beziehungsweise 512 (Varying);
- autoregressives Validation-Expert-Matching alle 25 Updates ab Update 0;
- keine ergebnisabhängigen Stopps, kein Finetuning und kein Hard Thresholding;
- ausschließlich Separate-Channel-Gruppen und native Masken.

Der Umfang beträgt damit 30 GO- und sechs GR-Produktionsruns. Pro Protokoll
laufen 15 GO- und drei GR-Worker unabhängig und vollständig parallel.

## Persistenz und Resume

Das gemeinsame Studienmodul `GO_Sensitivity/Package6Study.jl` besitzt Grid,
GR-Werte, Seedplan, Budgets, kurze Run-IDs und Ergebnisstatus. Beispiele:

```text
results/study/fixed/go/s01/r01
results/study/varying/go/s05/r03
results/study/fixed/gr/r01
```

Jeder Trainingsworker persistiert atomar:

- vollständige Konfiguration und Fingerprint;
- initialen Apprentice-Parameterhash;
- `running`, `complete` oder `failed`;
- alle Evaluation-Shards und Masken;
- das bestehende Pareto-Modellarchiv;
- `resume/latest.jld2`;
- Laufzeit- und Abschlusszusammenfassung.

Ein vollständiger Run wird nach Identitätsprüfung übersprungen. Ein laufender
Run wird aus dem letzten Resume-Checkpoint fortgesetzt. Ein fehlgeschlagener
Run wird nicht stillschweigend ersetzt; nach Behebung der Ursache ist ein
expliziter Neustart mit `--retry-failed` erforderlich.

Das technische `ParetoArchive` verwaltet Modelle weiterhin kompatibel anhand
aktiver globaler Eingänge. Die wissenschaftliche Paket-6-Auswertung berechnet
ihre Fronten unabhängig davon ausdrücklich auf
`(aktive SC-Gruppen, Validation-MSE)`.

## Launcher und Analyseworker

`GO_Sensitivity/launch_study_tmux.sh` unterstützt:

- `--protocol all|fixed|varying`;
- `--preview`;
- `--analysis-only`;
- `--results-dir PATH`.

Der Standardstart erzeugt 36 Trainingssessions und gleichzeitig zwei
Analyse-/Wartesessions. Ein Einzelprotokoll erzeugt 18 Trainingssessions und
eine Analysesession. Alle tmux-Sessions schließen sich nach Prozessende selbst.
Logs, `jobs.tsv`, Launch-Metadaten und das JLD2-Studienmanifest liegen unter
`results/study/launches/<launch-id>/`.

Der Fixed- beziehungsweise Varying-Analyseworker pollt alle 60 Sekunden für
höchstens 14 Tage. Er bricht bei einem explizit fehlgeschlagenen Run mit
`failure_report.md` ab und kann nach Reparatur beziehungsweise Neustart des
Trainingsworkers erneut gestartet werden.

## Audit

Vor jeder Auswertung verlangt der Analyseworker exakt 15 GO- und drei GR-Runs
des Protokolls und prüft:

- Run- und Config-Fingerprints;
- Expert- und Corpus-Identitäten;
- SC-only, native Sparsity, Strengths, Lernrate und Budget;
- vollständige Evaluation-Abdeckung von Update 0 bis zum Endbudget;
- Apprentice- und Batchseed-Paarung;
- identische initiale Parameterhashes innerhalb jedes Replicates;
- verschiedene Seeds zwischen den Replicates;
- numerischen Status und Abschlusszusammenfassung.

## Offline-Metriken

Der Analyseworker berechnet für GO und GR:

- Fronten pro Run, pro Strength und global;
- empirische Attainment-Grenzen für 1/3, 2/3 und 3/3 Seeds über
  Gruppenzahlen `0:96`;
- erste Hitting Times, Validation-MSE und Reachability für jede Gruppenzahl,
  hervorgehoben bei `48`, `24`, `12`, `6`, `3` und `1`;
- Front-Regret zur finalen eigenen und zur gepoolten Strength-Front;
- Frontnähe als MSE von höchstens `1.10 ×` Front-Envelope bei gleicher oder
  höherer Sparsity;
- Late-Training-Fenster von 10 %, 20 % (primär) und 30 %;
- frontnahe Belegung, medianen und 90%-Regret, Exkursionen, Recovery-Zeiten
  und ungelöste Endexkursionen;
- Gruppen-, MSE- und gemeinsame Reset-Ereignisse einschließlich Raten pro
  1.000 Updates und Sprungamplituden;
- monotone Archivkonvergenz und Updates bis 90 % beziehungsweise 100 % der
  finalen Front-Envelope-Abdeckung;
- deskriptive Strength-Trends und Spearman-Rangkorrelation innerhalb jedes
  gepaarten Seeds, ohne Signifikanzüberhöhung bei drei Seeds;
- Jaccard-Ähnlichkeit globaler Sensor-Kanal-Masken bei exakt gemeinsamen
  Front-Gruppenzahlen;
- Auswahlhäufigkeiten der pro Run sparsesten Maske mit Validation-MSE
  `≤ 0.01`;
- numerische Fehlläufe, Laufzeiten, Frontgrößen, beste MSE-Punkte und
  sparseste archivierte Punkte.

GR wird in sämtlichen Offline-Metriken als Referenz geführt, nimmt aber nicht
an der Testkandidatenauswahl teil.

## Validation-basierte Kandidatenauswahl und terminaler Test

Die Testauswahl erfolgt ausschließlich aus der gepoolten nativen GO-Front:

- `C_match`: kleinster Validation-MSE;
- `C_sparse`: wenigste aktive SC-Gruppen unter allen Frontpunkten mit
  Validation-MSE `≤ 0.01`;
- Tie-Breaker: niedrigerer MSE, früheres Update, lexikographische Run-ID;
- sind beide Kandidaten identisch oder existiert kein zusätzlicher
  qualifizierter Punkt, wird nur `C_match` getestet.

Das Kandidatenmanifest wird atomar und unveränderlich geschrieben, bevor der
erste Rollout startet. Ein wiederholter Analyseaufruf verwendet dasselbe
Manifest; eine abweichende neu berechnete Auswahl ist ein Fehler.

Fixed verwendet die gemeinsame 200-Schritt-Testepisode. Varying verwendet die
acht bestehenden Testepisoden. Nur `C_match`, gegebenenfalls `C_sparse`, und
der Expert werden ausgerollt. Das Testset ist ein terminaler Funktionscheck:
Es beeinflusst weder Training noch Strength- oder Kandidatenauswahl. Seine
Ergebnisse lösen keine nachträgliche Entscheidung aus und sind damit für
dieses festgelegte Paket-6-Protokoll valide.

## Artefakte

Pro Protokoll entstehen:

- kompakter Pareto-SVG mit Seed- und gepoolten Fronten sowie
  Attainment-Grenzen;
- ausgedünnte Strength-/Seed-Trajektorien für aktive Gruppen und Log-MSE;
- Archivkonvergenz-, Frontnähe-/Exkursions- und Hitting-Time-Plots;
- Masken-Jaccard-Heatmaps und Sensor-/Kanalauswahlkarten;
- interaktiver 3D-HTML-Plot über Gruppen, Log-MSE und Updates;
- Test-Rewardkurven und für Varying ein Return-Boxplot;
- vollständige CSV- und JLD2-Metriken;
- das eingefrorene Kandidatenmanifest;
- ein ausführliches englisches `report.md` mit Audit, Tabellen,
  Kandidatenprovenienz, Plotlinks und Testresultaten.

Die Rohdaten bleiben vollständig erhalten. Die Paper-nahen SVGs zeigen nur
Fronten und aggregierte Kurven. Diagnostische Trajektorien werden gleichmäßig
ausgedünnt; Front- und Resetpunkte bleiben zwingend erhalten.

## Abnahme

Die Implementierung enthält Tests für Jobmanifest, Seedpaarung, kurze Pfade,
Dominanz, Fronten, Attainment, Regret, Frontnähe, Exkursionen, Recovery,
Archivabdeckung, Jaccard, Kandidatenauswahl, Workerstatus, Timeout,
Manifest-Unveränderlichkeit und Plot-/Persistenz-Smoke-Tests. Der Launcher
liefert im Preview 38 Sessions für `all` und 19 pro Einzelprotokoll.

Paket 6 ist abgeschlossen, wenn beide Protokollreports den Audit bestehen,
die terminalen Testchecks ohne Auswahlrückwirkung vorliegen und alle
maschinenlesbaren Artefakte reproduzierbar erzeugt wurden.
