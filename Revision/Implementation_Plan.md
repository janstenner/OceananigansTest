# Implementation Plan for the Paper Revision

Stand: 2026-09-01

Dieser Plan enthält nur ganze, in sinnvoller Reihenfolge abzuarbeitende Implementierungs- und Experimentpakete.
Nur die Paketüberschriften sind abhakbar.
Details zur konkreten technischen Umsetzung werden während der Bearbeitung im jeweiligen Paket festgelegt.

Nicht Teil dieses Plans sind ein Reward-Modul oder Reward-Estimator-Training, zusätzliche Ra/Pr-Regime, neue Sensor- oder Aktuatorlayouts, 3D RBC, andere Strömungsprobleme, Hardwareexperimente und zusätzliche RL-Algorithmen.

- [x] **Paket 1 — Neue Varying-IC-Erzeugung**

  Umfang:

  - Direkt einlesbare Corpus-Implementierung unter `Revision/VaryingIC_Corpus`.
  - JLD2-Corpus mit 20 Training-, 1 Validation- und 2 Test-Basiszuständen.
  - Unabhängige Seeds und strikte Trennung der Basen zwischen den drei Splits.
  - Reproduzierbares Sampling mit übergebenem RNG, horizontaler Spiegelung und periodischem Offset.
  - Korrekte Behandlung der Oceananigans-Interior-Felder und des horizontal face-centered $u$-Feldes.
  - Übersichtsplots für Training, Validation und Test.
  - Keine automatische Generierung beim Einlesen der Corpus-Datei.

  Abschluss:

  - Corpus-Erzeugung, Transformationen, JLD2-Persistenz, Plotting und Wiedereinsetzen der Felder in Oceananigans sind geprüft.

- [x] **Paket 2 — Generelle Fixed-IC- und Varying-IC-Run-Files für MAT und IPPO**

  Umfang:

  - Vier eigenständige Run-Files unter `Revision/Run_Files` für MAT und parameter-sharing IPPO unter Fixed IC und Varying IC.
  - Gemeinsame physikalische und numerische Konfiguration; protokollspezifische Modellgrößen aus den jeweiligen Fixed-IC- und Varying-IC-Ausgangsdateien bleiben erhalten.
  - Reproduzierbare, von übergeordneten Runnern über `REVISION_RUN_SEED` gesetzte Seeds; manuelle Läufe behalten einen zufälligen Fallback sowie benannte, getrennte Standard-Ausgabeverzeichnisse.
  - JLD2-Speicher- und Laderoutinen für Agent und Hook nach dem Muster der Originaldateien.
  - Trainingsverläufe, Laufzeiten, zusätzliche Metadaten und die konkrete Experiment-Checkpointplanung werden von späteren übergeordneten Experimentdateien organisiert.
  - Die Varying-IC-Run-Files inkludieren `VaryingICCorpus.jl`; `generate_random_init` unterstützt Split, RNG, Basis-Seed, Spiegelung und Offset und gibt `(result, split, base_seed, mirror, offset)` zurück.
  - Die konkrete Verwendung von Training-, Validation- und Test-Split wird von den späteren übergeordneten Experimentdateien festgelegt.

  Abschluss:

  - Alle vier Kombinationen lassen sich laden und initialisieren; Fixed-IC-Reset, Corpus-basierter Varying-IC-Reset sowie nummeriertes Speichern und Laden von Agent und Hook sind geprüft.

- [ ] **Paket 3 — MAT-Stabilitätsstudie**

  Umfang:

  - Vergleich von `python_like` mit gemeinsamer Value-Chain, LayerNorm,
    Self-Attention vor Cross-Attention und autoregressiver Rückführung der
    Actions.
  - Vergleich von `modified_half` mit separater Value-Chain, ohne LayerNorm,
    Cross-Attention vor Self-Attention und weiterhin autoregressiver
    Rückführung der Actions.
  - Vergleich von `modified_full` mit den Änderungen aus `modified_half` und
    zusätzlicher autoregressiver Rückführung der vorhergesagten Mittelwerte.
  - Revidierte MAT-Defaults mit separater Value-Chain und autoregressiver Rückführung vorhergesagter Mittelwerte; LayerNorm und Self-Attention vor Cross-Attention sind standardmäßig deaktivierte, konfigurierbare Varianten, und sämtliche Legacy-Dropout-Defaults stehen auf `0.0`.
  - Fünf gepaarte Runs pro Konfiguration unter Fixed IC mit jeweils exakt
    2.000 Episoden sowie fünf gepaarte Runs pro Konfiguration unter Varying IC
    mit jeweils exakt 4.000 Episoden.
  - Identische Netzinitialisierung der jeweils vergleichbaren Komponenten und
    identische IC-Auswahlfolgen innerhalb jedes Replikats.
  - Selektiver automatischer Start der Fixed-IC-, Varying-IC- oder aller zehn
    detached tmux-Worker, atomare JLD2-Speicherung nach jeder Konfiguration und
    restart-sichere, beschreibende Ergebnispfade ohne manuelle Laufnummern.
  - Inkrementelle Sammlung aller jeweils vorhandenen gültigen Ergebnisdateien
    mit Paarungsprüfung, maschinenlesbaren Kennzahlen sowie den aus dem
    aktuellen Datenstand möglichen Lernkurven-, Performance- und
    Laufzeitplots; fehlende Protokolle oder Runs sind dabei zulässig.
  - Auswertung von Trainingsstabilität, Fehlläufen, Lernkurven, finaler Performance, Streuung und Laufzeit.
  - Keine gesonderte Dropout-Behauptung oder Dropout-Ablation.

  Implementierungsstand:

  - Worker, Seedplan, Initialisierungs- und IC-Paarungsprüfungen,
    tmux-Launcher, JLD2-Schema und Collector liegen unter
    `Revision/MAT_Stability`.
  - Null-Episoden-Prüfungen für Fixed IC und Varying IC sowie kurze echte
    Episodenläufe prüfen die technische Pipeline vor dem Serverlauf.
  - Das Paket bleibt bis zur Erzeugung und Auswertung aller 30 Produktionsruns
    offen.

  Abschluss:

  - Die Stabilitätswirkung der Änderungen vor und nach Einführung des Mean Tokens ist mit gepaarten, aggregierten RBC-Ergebnissen belegt.

- [ ] **Paket 4 — MAT versus IPPO unter Fixed IC und Varying IC**

  Detailplanung: `Revision/package4.md`

  Umfang:

  - Neudurchführung des MAT-IPPO-Vergleichs unter Fixed IC.
  - Neudurchführung des MAT-IPPO-Vergleichs unter dem neuen Varying-IC-Protokoll.
  - Identische Budgets, Seedpläne und Auswertungsmetriken innerhalb jedes Vergleichs.
  - Checkpointauswahl ohne Zugriff auf die finalen Varying-IC-Testbasen.
  - Aggregierte Lernkurven, finale Performance, Streuung, Fehlläufe und Laufzeiten.

  Implementierungsstand:

  - Seed-/IC-Plan, atomare Worker, Paket-3-Importer, begrenzter persistenter
    tmux-Launcher, deterministische Validation und inkrementeller Collector
    liegen unter `Revision/MAT_IPPO_Comparison`.
  - Null-Episoden-Initialisierungen und vollständige Validation-Rollouts prüfen
    alle vier Fixed/Varying- und MAT/IPPO-Kombinationen vor dem Serverlauf.
  - Das Paket bleibt bis zur Erzeugung und Auswertung der Produktionsruns offen.

  Abschluss:

  - Beide im Paper verwendeten MAT-IPPO-Vergleiche sind mit den neuen Run-Files vollständig reproduziert.

- [x] **Paket 5 — Window-Size-Experiment — GESTRICHEN**

  Status:

  - Dieses Experiment wurde bewusst aus dem Revisionsumfang entfernt und wird
    nicht durchgeführt. Die Paketnummer und der ursprünglich vorgesehene
    Umfang bleiben zur Dokumentation der Scope-Entscheidung erhalten.

  Ursprünglich vorgesehener Umfang — entfällt:

  - ~~Neudurchführung aller Window-Size-Konfigurationen, die im Paper verbleiben.~~
  - ~~Einheitliche Trainings- und Auswertungsbedingungen über alle Fenstergrößen.~~
  - ~~Speicherung der Performance-, Stabilitäts- und Laufzeitergebnisse.~~

  Entscheidung:

  - Es sind keine Implementierung, Produktionsruns oder Auswertung für das
    Window-Size-Experiment mehr erforderlich.

- [ ] **Paket 6 — Minimale GO-Sensitivitätsstudie auf RBC-Daten**

  Umfang:

  - Keine Toy-Studie, sondern eine kleine kontrollierte Sensitivitätsstudie direkt auf den für die Apprentice-Distillation verwendeten RBC-Daten.
  - Variation der wesentlichen GO-Regularisierungseinstellung in einem kleinen vorab festgelegten Bereich.
  - Wiederholung über mehrere Apprentice-Seeds.
  - Auswertung von Validation-Expert-Matching, aktiven SC-Gruppen,
    Trainingsstabilität, Maskenstabilität und numerischen Fehlläufen.
  - Vergleich mit GR unter demselben Auswertungsprotokoll.
  - Der wissenschaftliche Paket-6-Sweep verwendet ausschließlich
    Separate-Channel-Gruppierung. Channel-Coupled, Thresholding und weitere
    Regularisierer folgen in Paket 7/8.

  Implementierungsstand:

  - Die gemeinsame Paket-6/7/8-Infrastruktur unter
    `Revision/Expert_Apprentice_Distillation` enthält die Revision-Kopie des
    Expert-Apprentice-Codes, direkt inkludbares Corpus-Laden, atomare
    Fixed-/Varying-IC-Worker, deterministische Expertauflösung und einen
    persistenten tmux-Launcher.
  - Der Varying-IC-Plan erzeugt 40 Training-Worker aus 20 Basiszuständen und
    zwei Spiegelungen; jeder Training-Worker besitzt alle 96 Offsets.
    Validation und Test verwenden pro Basis-/Spiegelungskombination nur die
    festen Offsets 0 und 20, also vier beziehungsweise acht Episoden. Fixed IC
    verwendet einen Worker und eine Episode.
  - Die Worker speichern globale `3 × 48 × 8`-Sensortensoren statt mehrfach
    überlappender lokaler Fenster. Callback- und echte Ein-Schritt-Tests prüfen
    atomare Speicherung, Split-Merge und bitgenaue Rekonstruktion der
    `360 × 12` MAT-Observation.
  - Der Apprentice-Code verwendet nur noch die kanonischen Methoden `:go`,
    `:gr`, `:group_lasso` und `:growl`. Ein Trainingsschritt ist genau ein
    Optimizer-Update; Regularisierungsstärke, Proximalintervall und ein
    optionaler Finetune-Abschnitt mit fester nativer Maske sind explizit
    konfiguriert. Ergebnisabhängige Stopps, Rollback und zufällige
    Nullgruppenwiederherstellung sind entfernt.
  - Echte Corpus-Validation, reine gruppenkonsistente Maskenerzeugung und das
    gemeinsame Pareto-Archiv sind implementiert. Alle Messwerte und Masken
    bleiben erhalten; nur aktuelle Run-Fronten besitzen Modellcheckpoints,
    wobei alle Masken eines Trainingsschritts genau ein Modell teilen.
    Dominierte Modelle werden referenzbasiert periodisch und final bereinigt;
    der Restart-Checkpoint liegt getrennt unter `resume/latest.jld2`.
  - Paket 6 erzeugt ausschließlich native SC-Kandidaten. Hard-Threshold-
    Kandidaten werden erst in Paket 7/8 ergänzt.
  - Ein reiner Julia-Runner unter `Revision/GO_Sensitivity` führt lokal den
    Fixed-IC-Technikpilot mit Expert-/Corpus-Provenienzprüfung, Resume und
    getrennten nativen beziehungsweise Hard-Threshold-Pareto-Scopes aus. Der
    lokale Runner verwendet bewusst weder Bash noch tmux.
  - Der Plain-Julia-Runner `run_strength_calibration_pilot.jl` rechnet die
    vollständige bisherige Strength-Menge als homogenen Ein-Seed-Block neu:
    Fixed-GC `0.003/0.006/0.01/0.03/0.06/0.09`, Fixed-SC
    `0.0015/0.003/0.006/0.01/0.02/0.03` sowie Varying-GC und Varying-SC jeweils
    `0.003/0.008/0.025/0.04/0.06`. Jeder der 22 Fälle läuft wegen der
    include-time Konfiguration in einem eigenen frischen Julia-Prozess. Alle
    Läufe verwenden exakt dieselbe gepaarte Initialisierung und Batch-Reihenfolge,
    Regressions-Lernrate `2e-4`, 35.000 Fixed- beziehungsweise 50.000
    Varying-Updates. Phase, Lernrate und Budget gehören zur Run-Identität. Die
    Kalibrierung dient zur Festlegung von fünf Produktionsstärken je Protokoll
    und zählt nicht als wissenschaftlicher Paket-6-Sweep.
  - `GO_Sensitivity/launch_tmux.sh` startet standardmäßig alle 22
    Strength-Fälle gleichzeitig als je eine selbstbeendende tmux-Session mit
    eigenem Log. Abgeschlossene Runs werden übersprungen,
    unterbrochene Runs aus `resume/latest.jld2` fortgesetzt.
  - Ein eigener leichtgewichtiger Kalibrierungs-Inspector erzeugt für alle vier
    Protokoll-/Gruppierungskombinationen PlotlyJS-Pareto-Plots über aktive
    Gruppen und logarithmischen autoregressiven Validation-MSE. Alle
    Checkpoints bleiben unverbundene, nach Stärke eingefärbte Punkte; die
    gepoolten nichtdominierten Punkte werden zusätzlich markiert.
  - Der Kalibrierungs-Inspector akzeptiert ausschließlich vollständige
    homogene 35.000/50.000-Update-Blöcke mit Lernrate `2e-4` und kann anschließend alle in deren
    stärkeweisen Pareto-Archiven erhaltenen Kandidaten in echten 200-Schritt-
    Episoden auswerten. Fixed verwendet die eine gemeinsame Episode; Varying
    verwendet alle acht festgelegten Testepisoden. Pro Kombination entstehen
    Expert-/Apprentice-Rewardkurven, ein Return-Boxplot, CSV und JLD2; einzelne
    Episoden werden identitätsgeprüft gecacht. Diese Auswertung ist als
    `calibration_test_diagnostic` gekennzeichnet und darf keine Stärke oder
    Kandidatenauswahl bestimmen.
  - Die technische Kalibrierung bleibt interne Vorarbeit und wird im Paper
    nicht erwähnt. Ihre Testdiagnostik traf keine Auswahlentscheidung; die
    Produktionsauswahl von Paket 6 wird ausschließlich anhand des
    Validation-Expert-Matchings und der nativen SC-Sparsity eingefroren.
  - Ein leichtgewichtiger Fixed-Pilot-Inspector gibt Trainingsmetadaten, den
    vollständigen nativen GO-Verlauf, den finalen Thresholdvergleich und das
    erhaltene Pareto-Archiv aus und exportiert diese Daten als Text, CSV und
    PlotlyJS-Pareto-Plot über aktive Gruppen und autoregressiven
    Validation-MSE, ohne Expert-, Apprentice- oder Resume-Checkpoints zu
    deserialisieren. Der verlängerte Technikpilot verwendet 6.000 Updates; die
    Validation beginnt bei Update 2.000 und erfolgt danach alle 5 Updates.
  - Der Fixed-IC-Pilot-Inspector kann zusätzlich den bestmatchenden nativen
    Archivkandidaten und den MAT-Expert deterministisch über je eine echte
    200-Schritt-Episode auswerten. Reward-, globale Nusselt- und Aktionskurven
    werden identitätsgeprüft als JLD2 gecacht; Rewardvergleich, CSV und
    PlotlyJS-Kurve werden im Analyseordner gespeichert. Dies ist ein technischer
    Einzelvergleich und kein Closed-Loop-Sweep über Sensitivitätspunkte.
  - Die Produktionsstärken sind für Fixed und Varying auf
    `0.0015/0.003/0.006/0.01/0.03` festgelegt. Drei neue Seedpaare werden aus
    `StableRNG(20260810)` erzeugt; GO und GR teilen innerhalb eines Replicates
    Initialisierung und Batchreihenfolge. GR verwendet `0.00004` unter Fixed
    und `0.0001` unter Varying.
  - `Package6Study.jl` und `run_study_worker.jl` implementieren die 30 GO- und
    sechs GR-Produktionsruns mit kurzen Pfaden, atomarem Status,
    Konfigurationsfingerprint, initialem Parameterhash, Resume und expliziter
    Fehlerbehandlung. Fixed läuft 35.000, Varying 50.000 Updates; beide nutzen
    Lernrate `2e-4` und Validation alle 25 Updates ab Update 0.
  - `launch_study_tmux.sh` startet standardmäßig 36 Trainings- und zwei
    Analyse-/Wartesessions gleichzeitig. Er unterstützt Protokollauswahl,
    Preview, Analysis-only und Result-Root; Launchlogs und maschinenlesbare Manifeste liegen unter
    `results/study/launches/<launch-id>/`.
  - Der protokollspezifische Analyseworker auditiert jeweils 15 GO- und drei
    GR-Runs, berechnet Run-/Strength-/globale Fronten, Attainment, Hitting
    Times, Regret, Frontnähe, Exkursionen, Recovery, Resets,
    Archivkonvergenz, Strength-Trends und Maskenstabilität und erzeugt kompakte
    SVGs, einen interaktiven 3D-Plot, CSV/JLD2 und einen englischen Report.
  - `C_match` und gegebenenfalls `C_sparse` werden ausschließlich aus der
    gepoolten nativen GO-Validation-Front ausgewählt und vor jedem Testrollout
    atomar eingefroren. Fixed verwendet danach die gemeinsame 200-Schritt-
    Testepisode, Varying die acht bestehenden Testepisoden. Nur die
    eingefrorenen GO-Kandidaten und der Expert werden getestet; Testergebnisse
    verändern keine Auswahl oder Trainingsentscheidung.
  - Synthetische Metrik-, Pairing-, Status-, Timeout-, Kandidatenmanifest- und
    Plot-Smoke-Tests sind implementiert. Der Launcher-Preview wurde mit 38
    Sessions für beide beziehungsweise 19 für ein Protokoll verifiziert.

  Abschluss:

  - Es liegt genügend RBC-basierte Evidenz vor, um GO entweder mit einer begründeten Konfiguration im Paper zu behalten oder als sekundär einzustufen.

- [x] **Gemeinsame Expert- und Unactuated-Testbaselines**

  - Ein Fixed- und acht Varying-Testrollouts mit den veröffentlichten Experts.
  - Dieselben Testfälle zusätzlich mit durchgehend null gesetzten Actions.
  - Vier parallel startbare, atomare JLD2-Referenzdateien mit Reward,
    vollständigem `state_Nu`, Actions, aggregierten Scores und Provenienz.

- [ ] **Paket 7 — Apprentice Distillation unter Fixed IC mit Pareto-Set-Erzeugung**

  Umfang:

  - Neudurchführung der Fixed-IC-Apprentice-Trainings für die im Paper verbleibenden Regularisierer und Channel-Gruppierungen.
  - Fixed-IC-Training-, Validation- und Testdaten sind identisch.
  - Regelmäßige Apprentice-Checkpoints bis zu einem festen Maximalbudget.
  - Trennung von Apprentice-Checkpointauswahl und anschließendem Hard Thresholding beziehungsweise Mask Extraction.
  - Pareto-Kandidaten aus allen vereinbarten Apprentice-Seeds und Checkpoints.
  - Die beiden Pareto-Messgrößen sind die Gesamtzahl aktiver Inputs und das Expert Matching auf dem Validation Set.
  - Dominierte Kandidaten werden verworfen. Die zu nichtdominierten Punkten gehörenden Apprentice-Modelle und Masken werden gespeichert.
  - Closed-Loop-Simulationen werden auf eine kleine Auswahl aus dem Pareto Set begrenzt.
  - Neudurchführung der Fixed-IC-Vergleiche, Sparsity-Auswertungen, Sensorplots und Tabellen des Papers.

  Abschluss:

  - Das finale Fixed-IC-Modell folgt aus einem gespeicherten und reproduzierbaren Pareto Set statt aus manuellem Stoppen.

- [ ] **Paket 8 — Apprentice Distillation unter Varying IC mit Pareto-Set-Erzeugung**

  Umfang:

  - Neudurchführung der Varying-IC-Apprentice-Trainings für die im Paper verbleibenden Regularisierer und Channel-Gruppierungen.
  - Training-, Validation- und Testdaten stammen strikt aus den jeweils getrennten Corpus-Splits.
  - Speicherung von Basis-Seed, Spiegelung, Offset, Run-Seed, Episode und Zeitschritt in allen Rollout-Daten.
  - Regelmäßige Apprentice-Checkpoints bis zu einem festen Maximalbudget.
  - Trennung von Apprentice-Checkpointauswahl und anschließendem Hard Thresholding beziehungsweise Mask Extraction.
  - Pareto-Kandidaten aus allen vereinbarten Apprentice-Seeds und Checkpoints.
  - Die beiden Pareto-Messgrößen sind die Gesamtzahl aktiver Inputs und das Expert Matching auf dem Validation Set.
  - Dominierte Kandidaten werden verworfen. Die zu nichtdominierten Punkten gehörenden Apprentice-Modelle und Masken werden gespeichert.
  - Die Testbasen werden erst nach Abschluss der Modell- und Maskenauswahl verwendet.
  - Closed-Loop-Simulationen werden auf eine kleine Auswahl aus dem Pareto Set und anschließend auf die finalen Testmodelle begrenzt.
  - Neudurchführung der Varying-IC-Vergleiche, Sparsity-Auswertungen, Sensorplots und Tabellen des Papers.

  Abschluss:

  - Das finale Varying-IC-Modell folgt aus einem gespeicherten, leckagefreien und reproduzierbaren Pareto Set.

- [x] **Paket 9 — Baselines bei gleicher Sensorzahl — GESTRICHEN**

  Status:

  - Dieses Experiment wurde bewusst aus dem Revisionsumfang entfernt und wird
    nicht durchgeführt. Die Paketnummer und der ursprünglich vorgesehene
    Umfang bleiben zur Dokumentation der Scope-Entscheidung erhalten.

  Ursprünglich vorgesehener Umfang — entfällt:

  - ~~Random-Mask-Baseline mit derselben Gesamtzahl aktiver Inputs wie das ausgewählte Sparse-Modell.~~
  - ~~Uniform beziehungsweise geometrisch gleichmäßig verteilte Maske mit derselben Gesamtzahl aktiver Inputs.~~
  - ~~Offline-Screening der Random Masks anhand des Expert Matchings.~~
  - ~~Apprentice-Training und Closed-Loop-Vergleich unter denselben Daten-, Seed- und Testbedingungen.~~
  - ~~Lasso und Standard-GrOWL werden nur dann als Paper-Baselines geführt, wenn sie mit demselben Pareto- und Performanceprotokoll ausgewertet werden.~~

  Entscheidung:

  - Es sind keine Implementierung, Produktionsruns oder Auswertung für die
    Baselines bei gleicher Sensorzahl mehr erforderlich.

- [ ] **Paket 10 — Test unter Sensorrauschen**

  Detailplanung: `Revision/package10.md`

  Umfang:

  - Post-hoc-Sensorrauschen ausschließlich während der Anwendung und ohne Retraining.
  - Weißes additives Gaußrauschen mit den gemeinsamen festen Rauschstufen
    `0.0/0.01/0.05/0.10/0.20/0.30/0.40/0.50/0.70/1.00` relativ zu protokollspezifischen kanalweisen
    Datenskalen.
  - Auswertung des finalen dichten Experts, des validation-only sparsesten
    SC-Apprentices und des Paket-6-`C_match`-Kandidaten auf denselben Testfällen
    und gepaarten Rauschrealisierungen.
  - Sparse-Auswahl ausschließlich unter den eingefrorenen SC-Kandidaten durch
    minimale `active_inputs` und danach minimale Validation-MSE bei Gleichstand;
    damit Fixed `go-sc` und Varying `gr-sc`.
  - Zehn Rauschreplikate pro nichtverschwindendem Level und Testfall; Level
    `0.0` importiert die vorhandenen sauberen Baselines ohne Replikate.
  - Getrennte Ergebnisse für Fixed IC und Varying IC.
  - Vergleich des Performanceabfalls relativ zur jeweils ungestörten Auswertung.

  Implementierungsstand:

  - `Revision/Noise_Study` enthält die eingefrorene Manifest- und
    Kandidatenauflösung, exakte Trainingscorpus-Kanalskalen nach Entfernung der
    Positionskodierung, controller-unabhängige gepaarte Noise-Seeds, atomare
    Clean-/Noise-Worker und einen filterbaren restart-sicheren tmux-Launcher.
  - Ein Worker besitzt jeweils eine Kombination aus Protokoll, Controller und
    Rauschlevel und führt alle zugehörigen Fälle und Replikate hintereinander
    aus. Der vollständige Launch umfasst 60 selbstschließende Sessions; über
    wiederholte `--noise-level`-Argumente lassen sich insbesondere nur die
    Ergänzungslevel `0.30/0.40/0.50` beziehungsweise `0.70/1.00` starten.
  - Der erste Produktionslauncher startet bewusst keinen Analysis-Worker; die
    spätere Auswertung folgt als separates Julia-Skript.
  - `Noise_Study/make_paper_tables.jl` erzeugt unabhängig vom Launcher eine
    CSV- und Markdown-Ergebnistabelle über Controller, Rauschlevel und mittleres
    Testset-`state_Nu`. Ohne Argument verwendet es je Protokoll die neueste
    Experiment-ID; unterschiedliche IDs führen nur zu einer Warnung, und noch
    fehlende Workerresultate werden als `NA` erhalten. Die Kandidaten erscheinen
    in der Reihenfolge Expert, `C_match`, Sparse.
  - Das Paket bleibt bis zu den Produktionsruns und ihrer separaten Auswertung
    offen.

  Abschluss:

  - Der Robustheitsverlust von Expert und Apprentice unter Sensorrauschen ist reproduzierbar und quantitativ vergleichbar.

- [ ] **Paket 11 — Additional Experiment: Direct RL Training on a Selected Sensor Set**

  Umfang:

  - Reproduzierbarer Runner für direktes RL auf einer bereits ausgewählten und danach festen Sensormaske.
  - Neudurchführung des Dense-versus-Selected-Sensor-Set-Vergleichs unter Fixed IC.
  - Neudurchführung des Dense-versus-Selected-Sensor-Set-Vergleichs unter Varying IC.
  - Die Maske wird in diesem Experiment weder ausgewählt noch während des RL-Trainings angepasst.
  - Identische Seedpläne und Testfälle innerhalb der Vergleiche.
  - Auswertung von Lernkurven, finaler Performance, Stabilität, Simulationsschritten, Datendurchsatz und Laufzeit.
  - Klare Trennung dieses Zusatzexperiments von der Expert-Apprentice-Distillationspipeline.

  Abschluss:

  - Beide im Paper gezeigten Masked-versus-Unmasked-Experimente sind vollständig neu durchgeführt und reproduzierbar.

- [ ] **Paket 12 — Vollständige Neuerstellung der Paper-Ergebnisse und Reproduzierbarkeitsaudit**

  Umfang:

  - Neuerstellung aller im Paper verbleibenden Abbildungen und Tabellen ausschließlich aus den neuen Revisionsergebnissen.
  - MAT-IPPO-Kurven für Fixed IC und Varying IC.
  - Window-Size-Ergebnisse, sofern sie im Paper verbleiben.
  - Fixed-IC- und Varying-IC-Expert-Apprentice-Vergleiche.
  - Pareto-Plots, Sparsity-Tabellen und Sensor-Pattern-Plots.
  - Baseline-, Sensorrauschen- und Direct-Sparse-RL-Ergebnisse.
  - Mittelwerte, Streuungen, Mediane, gepaarte Differenzen und Rohpunkte, soweit für das jeweilige Experiment vereinbart.
  - Maschinenlesbare Speicherung aller berichteten Zahlen.
  - Rückverfolgbarkeit jedes Ergebnisses zu Run-Konfiguration, Seeds, Checkpoint, Maske, Threshold und Git-Stand.
  - Prüfung auf Split-Leakage und unbeabsichtigte Nutzung der Varying-IC-Testbasen.

  Abschluss:

  - Jede im Paper berichtete Zahl und jedes Resultat-Artefakt lässt sich aus den gespeicherten Revision-Runs reproduzieren und anschließend in `Revision_Workpackages.md` dokumentieren.
