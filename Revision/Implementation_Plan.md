# Implementation Plan for the Paper Revision

Stand: 2026-07-30

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
  - Auswertung von Expert Matching, Gesamtzahl aktiver Inputs, Maskenstabilität und numerischen Fehlläufen.
  - Vergleich mit GR unter demselben Auswertungsprotokoll.
  - Prüfung der GO-Gruppensortierung, Rückzuordnung und channel-coupled Gruppierung als Teil der Studie.

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
  - Paket 6 erzeugt bereits die später benötigten Hard-Threshold-Kandidaten,
    wertet für seine GO-Sensitivitätsfrage jedoch ausschließlich native
    Regularisierer-Sparsity aus.
  - Ein reiner Julia-Runner unter `Revision/GO_Sensitivity` führt lokal den
    Fixed-IC-Technikpilot mit Expert-/Corpus-Provenienzprüfung, Resume und
    getrennten nativen beziehungsweise Hard-Threshold-Pareto-Scopes aus. Der
    lokale Runner verwendet bewusst weder Bash noch tmux.
  - Der Plain-Julia-Runner `run_strength_calibration_pilot.jl` erhält den
    abgeschlossenen gepaarten Ein-Seed-Baselineblock unverändert und startet
    ausschließlich eine additive Erweiterung. Neu hinzukommen Fixed-GC
    `0.003/0.006/0.06`, Fixed-SC `0.0015/0.003/0.006` sowie Varying-GC und
    Varying-SC jeweils `0.04/0.06`. Jede der vier Kombinationen läuft wegen der
    include-time Konfiguration in einem frischen Julia-Prozess; innerhalb
    einer Kombination teilen die neuen Stärken exakt dieselbe Initialisierung
    und Batch-Reihenfolge. Alle neuen Läufe verwenden Regressions-Lernrate
    `2e-4`; Fixed verwendet 9.000 und Varying 15.000 Updates. Phase, Lernrate
    und Budget gehören zur Run-Identität. Die Kalibrierung dient zur Festlegung
    von fünf Produktionsstärken je Protokoll und zählt nicht als
    wissenschaftlicher Paket-6-Sweep.
  - Ein eigener leichtgewichtiger Kalibrierungs-Inspector erzeugt für alle vier
    Protokoll-/Gruppierungskombinationen PlotlyJS-Pareto-Plots über aktive
    Gruppen und logarithmischen autoregressiven Validation-MSE. Alle
    Checkpoints bleiben unverbundene, nach Stärke eingefärbte Punkte; die
    gepoolten nichtdominierten Punkte werden zusätzlich markiert.
  - Der Kalibrierungs-Inspector kombiniert ausschließlich vollständige
    Baseline- und Erweiterungsblöcke und kann anschließend alle in deren
    stärkeweisen Pareto-Archiven erhaltenen Kandidaten in echten 200-Schritt-
    Episoden auswerten. Fixed verwendet die eine gemeinsame Episode; Varying
    verwendet alle acht festgelegten Testepisoden. Pro Kombination entstehen
    Expert-/Apprentice-Rewardkurven, ein Return-Boxplot, CSV und JLD2; einzelne
    Episoden werden identitätsgeprüft gecacht. Diese Auswertung ist als
    `calibration_test_diagnostic` gekennzeichnet und darf keine Stärke oder
    Kandidatenauswahl bestimmen.
  - Sobald die Varying-Testdiagnostik angesehen wurde, sind die derzeitigen
    acht Testepisoden nicht mehr unangetastet. Vor einer finalen Paket-8-
    Held-out-Aussage müssen deshalb neue Testbasen samt Expert-Rollouts erzeugt
    und eingefroren werden; alternativ ist der bisherige Split ausdrücklich
    nur als explorativer Diagnostiksplit zu berichten.
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
  - Fixed- und Varying-Corpus für Training, Validation und Test sowie finale
    Experts sind vorhanden. Die aus der Kalibrierung abgeleiteten fünf
    GO-Produktionsstärken, Thresholdstufen und numerischen Akzeptanzbereiche
    stehen noch aus. Das Apprentice-Trainingsbudget ist mit 9.000 Updates für
    Fixed IC und 15.000 Updates für Varying IC sowie einer Regressions-Lernrate
    von `2e-4` für Paket 6 bis 8 festgelegt.

  Abschluss:

  - Es liegt genügend RBC-basierte Evidenz vor, um GO entweder mit einer begründeten Konfiguration im Paper zu behalten oder als sekundär einzustufen.

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

- [ ] **Paket 9 — Baselines bei gleicher Sensorzahl**

  Umfang:

  - Random-Mask-Baseline mit derselben Gesamtzahl aktiver Inputs wie das ausgewählte Sparse-Modell.
  - Uniform beziehungsweise geometrisch gleichmäßig verteilte Maske mit derselben Gesamtzahl aktiver Inputs.
  - Offline-Screening der Random Masks anhand des Expert Matchings.
  - Apprentice-Training und Closed-Loop-Vergleich unter denselben Daten-, Seed- und Testbedingungen.
  - Lasso und Standard-GrOWL werden nur dann als Paper-Baselines geführt, wenn sie mit demselben Pareto- und Performanceprotokoll ausgewertet werden.

  Abschluss:

  - Die finale Sensorselektion ist gegen günstige, gleich große Masken und alle im Paper verbleibenden Sparsity-Baselines verglichen.

- [ ] **Paket 10 — Test unter Sensorrauschen**

  Umfang:

  - Post-hoc-Sensorrauschen ausschließlich während der Anwendung und ohne Retraining.
  - Gemeinsame festgelegte Rauschstufen relativ zu den kanalweisen Datenskalen.
  - Auswertung des finalen dichten Experts und des finalen Sparse Apprentices auf denselben Testfällen und Rauschrealisierungen.
  - Getrennte Ergebnisse für Fixed IC und Varying IC.
  - Vergleich des Performanceabfalls relativ zur jeweils ungestörten Auswertung.

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
