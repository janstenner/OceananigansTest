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

  Umfang:

  - Neudurchführung des MAT-IPPO-Vergleichs unter Fixed IC.
  - Neudurchführung des MAT-IPPO-Vergleichs unter dem neuen Varying-IC-Protokoll.
  - Identische Budgets, Seedpläne und Auswertungsmetriken innerhalb jedes Vergleichs.
  - Checkpointauswahl ohne Zugriff auf die finalen Varying-IC-Testbasen.
  - Aggregierte Lernkurven, finale Performance, Streuung, Fehlläufe und Laufzeiten.

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
