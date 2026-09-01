# Paket 10 — Test unter Sensorrauschen

Stand: 2026-09-01

## Ziel und Abgrenzung

Paket 10 untersucht die terminale Closed-loop-Robustheit bereits eingefrorener
Controller unter additivem Sensorrauschen. Es findet kein Retraining, Fine-
Tuning und keine Modell-, Masken- oder Rauschlevelauswahl anhand der
Rauschresultate statt. Die spätere Analyse wird als separates Julia-Skript
ergänzt und ist nicht Teil des ersten Produktionslaunchers.

## Controller

Pro Protokoll werden genau drei Controller verglichen:

1. der veröffentlichte dichte MAT-Expert;
2. der finale sparse SC-Apprentice aus Paket 7 beziehungsweise 8;
3. der validation-only ausgewählte `C_match`-Kandidat aus Paket 6.

Die Sparse-Auswahl betrachtet ausschließlich die vier bereits eingefrorenen
SC-Kandidaten aus Paket 7/8. Zuerst wird `active_inputs` minimiert; nur bei
gleicher minimaler Sensorzahl entscheidet die kleinere Validation-MSE. Damit
ist Fixed auf `go-sc` und Varying auf `gr-sc` festgelegt. Testergebnisse gehen
nicht in diese Auswahl ein.

Aus Paket 6 wird ausschließlich die Rolle `C_match` aus dem eingefrorenen
protokollspezifischen `candidate_manifest.jld2` übernommen. `C_sparse` gehört
nicht zu Paket 10.

Jedes Noise-Study-Manifest speichert Auswahlregel, alle betrachteten SC-
Kandidaten, ausgewählte Kandidaten, Masken, lokale Checkpointpfade, SHA-256-
Identitäten und den Expert-Identifier.

## Rauschmodell und Kanalskalen

Das einzige Rauschmodell ist additives, mittelwertfreies, räumlich und zeitlich
unabhängiges Gaußrauschen:

```text
x_noisy[c, i, t] = x[c, i, t] + alpha * scale[c] * z[c, i, t]
z ~ N(0, 1)
```

Die festen Rauschlevel sind:

```text
(0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00)
```

Die drei `scale[c]` sind protokollspezifische Sample-Standardabweichungen der
physikalischen Kanäle `b`, `w` und `u` im vollständigen jeweiligen
Distillation-Trainingscorpus. Unter Fixed IC ist dies der gemeinsame
Trainingsworker; unter Varying IC sind es alle 40 Train-Shards. Die in den
Corpus-Observations enthaltene sinusförmige Positionskodierung wird vor der
Skalenberechnung aus Kanal 1 entfernt.

Während der Closed-loop-Auswertung wird pro Kontrollschritt zuerst genau ein
globales physikalisches `3 × 48 × 8`-Rauschfeld erzeugt. Erst danach wird die
rauschfreie Positionskodierung ergänzt und die überlappende `360 × 12`-MAT-
Observation rekonstruiert. Mehrfach vorkommende Fensteransichten desselben
Sensors erhalten dadurch denselben Messfehler. Bei Sparse- und `C_match`-
Apprentices wird die eingefrorene Maske nach der Rauschaddition angewendet;
inaktive Inputs bleiben exakt null.

Es gibt kein Clipping der Sensorwerte, keinen Bias, keinen Dropout und keine
zeitliche oder räumliche Korrelation.

## Paarung, Testfälle und Budgets

Der Rauschseed hängt ausschließlich von Protokoll, Rauschlevel, Replicate und
Testfallindex ab. Er enthält keinen Controller-Identifier. Alle drei
Controller erhalten damit dieselbe standardisierte Rauschfolge pro gepaartem
Fall.

- Fixed IC: ein bestehender 200-Schritte-Testfall, zehn Replikate pro
  nichtverschwindendem Rauschlevel.
- Varying IC: die acht bestehenden 200-Schritte-Testfälle, zehn Replikate pro
  Fall und nichtverschwindendem Rauschlevel.
- Rauschlevel `0.0`: keine Replikate; jeder Controller importiert seine bereits
  vorhandenen sauberen Testartefakte genau einmal.
- Alle Actions sind deterministische vorhergesagte Mittelwerte ohne
  zusätzliches Policy-Sampling.

## Worker- und Launcherdesign

`Revision/Noise_Study/launch_tmux.sh` startet einen Worker pro
`(Protokoll, Controller, Rauschlevel)`. Ohne Filter sind dies 60 selbst-
schließende tmux-Sessions.

Ein Worker besitzt seine vollständige Kombination:

- Clean: einen Baseline-Import über alle Fälle;
- Fixed noisy: zehn Episoden;
- Varying noisy: achtzig Episoden.

Episoden werden einzeln atomar gespeichert. Ein Restart prüft vorhandene
Episoden gegen Manifestfingerprint, Controller, Level, Replicate und Fall und
überspringt nur passende vollständige Dateien. `result.jld2` wird erst nach
vollständigem Workergrid erzeugt. Statusdateien unterscheiden `running`,
`complete` und `failed`; fehlgeschlagene Worker werden nur mit explizitem
`--retry-failed` erneut gestartet.

Der Launcher unterstützt Protokoll-, Controller- und Rauschlevelfilter,
Preview, bestehende Experiment-IDs, Threadlimits, aktive Session-Erkennung,
Launchlogs und maschinenlesbare Jobmanifeste. Ein Analysis-Worker wird in
diesem Schritt ausdrücklich nicht gestartet.

## Gespeicherte Episodendaten

Jede Episode enthält mindestens:

- Protokoll, Controller und vollständige Controller-/Manifestprovenienz;
- Rauschlevel, Kanalskalen, Replicate und Rauschseed;
- Testfall, Basis-Seed, Spiegelung und Offset soweit anwendbar;
- Reward, vollständiges `state_Nu`, Actions und Simulationszeit für alle 200
  Kontrollschritte;
- Summen/Mittelwerte und Action-Sättigungsanteil;
- atomare Completion-Metadaten.

Die spätere separate Analyse berechnet daraus den absoluten und relativen
Performanceverlust gegenüber dem jeweiligen sauberen Controller sowie die
gepaarte Differenz des Robustheitsverlusts zwischen Expert, Sparse und
`C_match`.
