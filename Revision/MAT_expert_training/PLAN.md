# MAT Expert Training — Versuchs- und Implementierungsplan

Stand: 2026-08-21

## Status

Die Trainings-, Stop-, Resume-, Test- und tmux-Infrastruktur ist implementiert.
Produktionsläufe wurden noch nicht gestartet.

## Ziel und wissenschaftliche Abgrenzung

Die Trainingsbudgets der MAT-Stabilitätsstudie und des MAT-IPPO-Vergleichs
bleiben fachlich angemessen für deren ursprüngliche Fragestellungen:

- Die MAT-Stabilitätsstudie vergleicht die drei MAT-Varianten bei einem
  einheitlichen festen Budget.
- Der MAT-IPPO-Vergleich misst MAT und parameter-sharing IPPO unter identischen
  festen Budgets und identischen Auswahlregeln.

Für die nachfolgenden Expert-Apprentice-, Sensorselektions- und
Robustheitsstudien wird dagegen nicht ein fairer Algorithmusvergleich, sondern
ein möglichst leistungsfähiger MAT-Teacher benötigt. Deshalb werden alle zehn
finalen MAT-Kandidaten unter Fixed IC und alle zehn unter Varying IC länger
trainiert. IPPO wird nicht weitertrainiert. Die breitere Parallelisierung
verkürzt die erwartete Zeit bis zum ersten Zielkandidaten.

Die verlängerten Läufe bilden ein eigenständiges Expert-Fine-Tuning-Experiment.
Sie verändern weder die Endpunkte noch die Auswertung der MAT-Stabilitätsstudie
oder des MAT-IPPO-Vergleichs und dürfen nicht rückwirkend in deren Lernkurven
oder Performancevergleich eingehen.

## Eingefrorene Ausgangskandidaten

Die Auswahl stammt aus
`Revision/MAT_IPPO_Comparison/results/analysis/mat_expert_ranking.csv`. Diese
Datei enthält zehn deterministisch validierte finale MAT-Checkpoints pro
Protokoll. Höhere, also weniger negative, Validation-Scores sind besser.

### Fixed IC

| Rang | Run-ID | Validation-Score | Run-Seed | IC-Seed | Herkunft |
|---:|---|---:|---:|---:|---|
| 1 | `seed_dfe17c7e95fcbb6d` | -590.9383601579968 | 1926005828 | 1313070640 | `generated` |
| 2 | `seed_92b79c49251eb7a2` | -593.0080725637902 | 1987317423 | 60237239 | `imported_package3` |
| 3 | `seed_14a5d7fe05cff1de` | -593.6025872180584 | 1838755672 | 181983109 | `generated` |
| 4 | `seed_1f82af6b2a587ef6` | -594.9301747976292 | 1241319044 | 83126739 | `imported_package3` |
| 5 | `seed_7ebdabd4b2738e12` | -597.169321125308 | 850985002 | 1210719576 | `generated` |
| 6 | `seed_e72249f3ea1fa410` | -598.6395111651227 | 1422047759 | 627765402 | `imported_package3` |
| 7 | `seed_5954219d6cc69d5c` | -600.2644632664014 | 459291457 | 409251705 | `generated` |
| 8 | `seed_ce0b5b582dda8eff` | -602.7203153676935 | 686791604 | 493568598 | `imported_package3` |
| 9 | `seed_4c43a7202fb90fbe` | -607.7752318311447 | 1301152156 | 512477356 | `generated` |
| 10 | `seed_3a2d3a2f3341b412` | -609.483440542703 | 319470045 | 1788533232 | `imported_package3` |

### Varying IC

| Rang | Run-ID | Validation-Score | Run-Seed | IC-Seed | Herkunft |
|---:|---|---:|---:|---:|---|
| 1 | `seed_5954219d6cc69d5c` | -619.8056356260331 | 459291457 | 409251705 | `generated` |
| 2 | `seed_e72249f3ea1fa410` | -621.5046799648103 | 1422047759 | 627765402 | `imported_package3` |
| 3 | `seed_14a5d7fe05cff1de` | -623.2918953664753 | 1838755672 | 181983109 | `generated` |
| 4 | `seed_92b79c49251eb7a2` | -625.3938140617716 | 1987317423 | 60237239 | `imported_package3` |
| 5 | `seed_dfe17c7e95fcbb6d` | -627.8491806615882 | 1926005828 | 1313070640 | `generated` |
| 6 | `seed_1f82af6b2a587ef6` | -628.7449545348579 | 1241319044 | 83126739 | `imported_package3` |
| 7 | `seed_ce0b5b582dda8eff` | -633.5548083921888 | 686791604 | 493568598 | `imported_package3` |
| 8 | `seed_3a2d3a2f3341b412` | -641.0630750222932 | 319470045 | 1788533232 | `imported_package3` |
| 9 | `seed_7ebdabd4b2738e12` | -646.0897190597743 | 850985002 | 1210719576 | `generated` |
| 10 | `seed_4c43a7202fb90fbe` | -646.9058766970634 | 1301152156 | 512477356 | `generated` |

Vor jedem Produktionsstart werden diese zwanzig Einträge gegen ihre Quelldateien
validiert; beim ersten nicht nur als Preview ausgeführten Start wird daraus ein
unveränderliches Kandidatenmanifest erzeugt. Es speichert zusätzlich den
SHA-256-Hash jedes Ausgangscheckpoints und der zugehörigen Validation-Datei.
Eine spätere Neuberechnung des Package-4-Collectors darf die bereits gestartete
Expert-Training-Auswahl nicht stillschweigend verändern.

## Resume-Einschätzung

Ein echtes Weitertrainieren der vorhandenen MAT-Endzustände ist grundsätzlich
möglich. Der in jeder Ergebnisdatei gespeicherte `agent` enthält nicht nur die
Netzgewichte, sondern auch:

- Encoder- und Decoder-Optimizer samt AdamW-Zuständen,
- den fortgeschrittenen Policy-`StableRNG`,
- den internen Updatezähler,
- die On-Policy-Trajectory,
- die für MAT benötigten letzten Aktions- und Value-Zustände.

Die Ergebnisdateien speichern außerdem Run- und IC-Seed, bisherige Rewards,
Fehlläufe, Hyperparameter, Quellhashes und unter Varying IC den vollständigen
Trainings-IC-Trace.

Der bisherige Package-4-Worker unterstützt dieses Resume trotzdem noch nicht
direkt: Er initialisiert für Training immer einen neuen Agenten, speichert den
Hook nicht und besitzt für Varying IC nur den ursprünglichen Plan über 4.000
Episoden. Die neue Codebasis implementiert deshalb folgenden kontrollierten
Resume-Pfad:

1. Run-File, MAT-Konfiguration, Parameterzahl, Quellidentität und
   Checkpointhash prüfen.
2. Environment und Hook frisch aufbauen, danach den vollständigen
   serialisierten Agenten einsetzen; die Optimizerzustände dürfen nicht neu
   initialisiert werden.
3. Bisherige Reward-, Fehler- und Laufzeithistorie aus der Ergebnisdatei in
   den neuen Runzustand übernehmen, ohne die Package-4-Quelldatei zu ändern.
4. Unter Varying IC den IC-Strom erneut aus `StableRNG(ic_seed)` erzeugen. Die
   ersten 4.000 Einträge müssen bitgenau dem gespeicherten Trace entsprechen;
   erst danach wird derselbe RNG-Strom für zusätzliche Episoden fortgesetzt.
5. Den gespeicherten Policy-RNG und den internen Updatezähler unverändert
   weiterverwenden. Der ursprüngliche `run_seed` wird als Identität geführt,
   nicht zum Neuinitialisieren der Policy benutzt.
6. Ab dem ersten neuen Checkpoint ein eigenes atomares Resume-Bundle speichern,
   das Agent samt Policy-RNG und Optimizerzuständen, Hook/Historie,
   Varying-IC-Fortsetzung, Zähler und Quellhashes enthält.

Der Resume-Audit validiert vor dem Produktionsstart die Quellidentitäten und
Hashes, den vollständigen Varying-IC-Präfix, Updatezähler und vorhandene
Optimizerzustände. Außerdem wurde die atomare Roundtrip-Serialisierung eines
vollständigen Agents einschließlich Optimizerzuständen geprüft.

## Experimentaufbau

Die Produktionsausführung besteht aus 21 parallelen tmux-Sessions:

- zehn voneinander unabhängigen Fixed-IC-Trainingsworkern;
- zehn voneinander unabhängigen Varying-IC-Trainingsworkern;
- einem Test- und Exportworker, der auf die beiden Gewinner und alle zwanzig finalen
  Trainingscheckpoints wartet und anschließend die beiden Gewinner auf dem
  jeweiligen Testset auswertet.

Jede Session schreibt ein eigenes Log, besitzt einen eindeutigen Lock, schließt
sich nach Prozessende selbst und kann anhand atomarer Status- und Resume-Dateien
neu gestartet werden.

## Testworker: Gewinner auf dem Testset

Der Testworker verändert keine Modelle und trifft keine Trainings- oder
Auswahlentscheidung. Er pollt den protokollweiten Gewinnerstatus. Ein
Protokoll wird erst ausgewertet, nachdem der Gewinner und auch die neun
anderen Worker ihre nach dem Stop-Signal noch laufenden Episoden beendet und
ihre finalen Checkpoints gespeichert haben.

### Testfälle

- Fixed IC: die eine gemeinsame vollständige 200-Schritt-Episode. Da Fixed IC
  keinen unabhängigen held-out Split besitzt, wird dieses Resultat als
  kontrollierter Fixed-IC-Vergleich und nicht als Generalisierungstest
  bezeichnet.
- Varying IC: alle acht vorab festgelegten Testepisoden aus zwei Testbasen,
  `mirror ∈ {false, true}` und `offset ∈ {0, 20}`. Jeder im Testmanifest
  enthaltene Fall wird genau einmal pro Controller vollständig über 200
  Kontrollschritte ausgerollt.

Der Testworker verwendet den finalen Gewinnercheckpoint mit deterministischen
Mean-Actions. Seine Ergebnisse dürfen weder die eingefrorene Auswahl noch
Trainingsziel oder Trainingsdauer bestimmen.

### Zu speichernde Analyseartefakte

Pro Gewinner und Testfall werden gespeichert:

- Protokoll, Controller-ID und Checkpoint-SHA-256,
- Basis-Seed, Spiegelung und Offset bei Varying IC,
- vollständige Rewardkurve und kumulativer 200-Schritt-Return,
- deterministische Auswertungsregel und Abschlussstatus.

Die zusammengefassten Artefakte umfassen CSV und JLD2 sowie SVG- und
PNG-Rewardkurven pro Protokoll. Einzelne Episoden werden atomar gecacht, sodass
ein Neustart nur fehlende oder ungültige Fälle berechnet.

Nach Abschluss beider Protokolltests werden die Trajectory-Puffer beider
Gewinner geleert und auf Kapazität eins verkleinert. Die kompakten Dateien
enthalten ausschließlich den Schlüssel `agent` und werden als
`results/experts/<protocol>/expert.jld2` gespeichert. Nach erfolgreichem
Reload beider Dateien ersetzen sie außerdem atomar pro Datei die versionierten
Distillation-Experts unter
`Expert_Apprentice_Distillation/experts/<protocol>/agent.jld2`.

## Zwanzig Trainingsworker

Jeder Worker besitzt genau einen der zwanzig Ausgangskandidaten und schreibt in
einen eigenen Ergebnisbaum. Originale Package-3-/Package-4-Dateien werden nur
gelesen und niemals überschrieben.

### Trainings- und Stopprotokoll

- Fixed IC setzt das Training auf derselben festen Anfangsbedingung fort.
- Varying IC setzt den deterministischen Trainings-IC-Strom nach dem
  verifizierten 4.000-Episoden-Präfix fort und verwendet weiterhin nur den
  Training-Split.
- Der Testsplit wird während des Trainings und der Gewinnerauswahl nicht
  geladen.
- Fixed IC: Nach jeder vollständig abgeschlossenen neuen Episode gewinnt der
  erste Worker mit einem Episodenreward strikt größer als `-555.0`.
- Varying IC: Nach jeder vollständig abgeschlossenen neuen Episode wird wie in
  `randomIC/randomIC_MAT.jl` der Mittelwert der letzten 100 Episodenrewards
  berechnet. Der erste Worker mit einem Mittelwert strikt größer als `-610.0`
  gewinnt.
- Die vorhandenen 4.000 Varying-Trainingsrewards werden beim Rolling-100-
  Fenster mitgeführt; die Prüfung erfolgt jedoch erst nach einer neuen,
  vollständig abgeschlossenen Episode.
- Unabhängig vom Threshold wird nach jedem atomaren Worker-Resume der globale
  Beststand des Protokolls aktualisiert: Fixed nach dem letzten Episodenreturn,
  Varying nach demselben Rolling-100-Mittel wie der Stopcheck. Auch die
  geladenen Ausgangs-/Resumezustände nehmen vor der ersten neuen Episode an
  diesem Vergleich teil.
- `results/<protocol>/best_so_far.jld2` ist ein vollständiger, atomar
  ersetzter Agentcheckpoint und bleibt deshalb auch bei Nichterreichen des
  Thresholds nutzbar.
- Parallel dazu wird `results/<protocol>/expert.jld2` aus einer separaten
  Agentkopie erzeugt. Die Datei enthält ausschließlich `agent`; dessen
  Trajectory ist wie beim finalen Expert-Export leer und auf Pufferkapazität
  eins verkleinert. Der aktive Trainingsagent bleibt unverändert.
- Der Gewinner veröffentlicht atomar ein protokollweites Stop-Signal. Andere
  Worker brechen keine laufende Episode ab. Sie beenden diese Episode,
  speichern Resume- und finalen Checkpoint und stoppen anschließend.
- Es gibt kein zusätzliches festes Episoden-Maximalbudget. NaN, Inf oder ein
  technischer Simulationsfehler führen zu einem expliziten Fehlerstatus.
- Ein bewusster vorzeitiger Cutoff wird über `finalize_best_so_far.jl`
  ausgelöst. Dabei wird der momentane globale Beststand unveränderlich als
  `manual_candidate.jld2` eingefroren; anschließend greifen dieselben
  Episode-Ende-, Finalcheckpoint-, Test- und Exportregeln wie beim
  Threshold-Gewinner.

### Checkpoints und Resultate

Jeder Worker speichert atomar:

- einen laufenden Resume-Status, einen finalen Abschlussstatus oder einen
  separaten Fehlerstatus,
- unveränderliche Ausgangsidentität und Parent-Checkpointhash,
- kumulatives und zusätzliches Episodenbudget,
- vollständige alte und neue Rewardhistorie mit klarer Fortsetzungsgrenze,
- den protokollspezifischen Stopwert und die Gewinneridentität,
- vollständiges Resume-Bundle,
- Laufzeiten, Fehlläufe, Konfiguration und Provenienz.

Ein abgebrochener Worker setzt ausschließlich den letzten passenden atomaren
Resume-Checkpoint fort. Abgeschlossene passende Worker werden übersprungen;
abweichende Konfigurationen oder Parent-Hashes verursachen einen Fehler und
werden nicht überschrieben.

## Kompakte Codebasis

Die Implementierung besitzt wenige klar getrennte Einstiegspunkte:

```text
Revision/MAT_expert_training/
├── PLAN.md
├── MATExpertTraining.jl
├── prepare_experiment.jl
├── run_training_worker.jl
├── run_test_worker.jl
├── finalize_best_so_far.jl
├── launch_tmux.sh
└── results/
```

`MATExpertTraining.jl` kapselt Kandidatenmanifest, Resume, IC-Plan,
Training/Stopkoordination, Testrollouts, atomare Persistenz und Ergebnisprüfung
gemeinsam. Fixed/Varying und die zwanzig Kandidaten werden nicht durch
duplizierte Workerimplementierungen abgebildet.

## Abnahmekriterien

- Das Kandidatenmanifest stimmt samt Scores, Seeds und SHA-256-Hashes mit dem
  vollständigen Package-4-Ranking überein.
- Der Resume-Audit bestätigt die deterministische Fortsetzung einschließlich
  Optimizer-, Policy-RNG-, Update- und Varying-IC-Zustand.
- Der Launcher plant genau 21 Sessions: zehn Fixed-, zehn Varying-
  Trainingsworker und einen wartenden Test-/Exportworker.
- Der Fixed-Gewinner erfüllt `episode_reward > -555.0`; der Varying-Gewinner
  erfüllt `mean(last_100_episode_rewards) > -610.0`.
- Solange noch kein Gewinner existiert, entspricht jeder protokollweite
  `best_so_far.jld2` dem höchsten bisher vollständig beobachteten Stopwert über
  alle zehn Worker; ein manueller Cutoff friert genau diesen Agentzustand ein.
- Zu jedem `best_so_far.jld2` existiert ein inhaltlich passender
  `expert.jld2`, der nur den kompaktierten Agenten mit leerer Trajectory
  enthält.
- Kein Worker wird mitten in einer Episode abgebrochen; alle zehn finalen
  Checkpoints pro Protokoll liegen vor dem Teststart vor.
- Der Testworker erzeugt Rewardkurven und Scores des Fixed-Gewinners auf der
  gemeinsamen Fixed-Episode und des Varying-Gewinners auf allen acht
  Varying-Testepisoden.
- Beide Gewinner werden als agent-only `expert.jld2` mit leerer, auf Kapazität
  eins verkleinerter Trajectory exportiert und ersetzen die beiden bestehenden
  versionierten Distillation-Experts.
- Kein Trainingsworker lädt den Varying-Testsplit.
- Alle Resultate sind restart-sicher, atomar und bis zum Ausgangscheckpoint,
  Seedplan und Quellstand rückverfolgbar.
- Das finale Expertmanifest wird ausschließlich aus dem eingefrorenen
  Trainingskriterium erzeugt; Testergebnisse haben keine Auswahlrückwirkung.
