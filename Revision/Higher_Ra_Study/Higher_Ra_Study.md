# Higher-Rayleigh-Number Study

## Scope

This directory will contain the complete expert-to-sparse-controller workflow
for the two higher-Rayleigh-number Varying-IC systems:

- `Ra = 5e4` (`ra5e4`);
- `Ra = 1e5` (`ra1e5`).

The final directory should be self-contained with respect to all
higher-Rayleigh-number study scripts and generated study data. Existing MAT
training results remain the immutable upstream source, while selected compact
experts, their test baselines, both distillation corpora, GO/GR training
artifacts, analyses, and final evaluation results live below
`Revision/Higher_Ra_Study`.

## Implemented first stage: MAT expert extraction

`analyze_runs.jl` performs the complete first stage for both Rayleigh
numbers:

1. Recursively scan the corresponding `MAT_expert_training` result root for
   available global best-so-far, completed final, and current resume
   checkpoints.
2. Require at least 100 completed episode rewards and recompute the mean of the
   latest 100 rewards for every candidate checkpoint.
3. Select the checkpoint with the largest rolling-100 mean independently for
   `ra5e4` and `ra1e5`.
4. Validate and byte-copy the matching compact `varying/expert.jld2` already
   published beside a selected best-so-far checkpoint. If no matching compact
   source exists, publish one from the selected checkpoint. The resulting
   `experts/<ra>/expert.jld2` contains only the agent; its trajectory is empty
   and every trajectory backing buffer has capacity one, matching the existing
   Revision distillation experts.
5. Save MAT-IPPO-style SVG learning curves for each Rayleigh number: a
   median/IQR aggregate with translucent individual runs and a separate
   individual-run figure. The displayed rolling window is 100 episodes, equal
   to the expert-selection metric.
6. Evaluate each selected expert deterministically on all eight cases of its
   own higher-Ra test split for 200 control steps. Results, reward histories,
   full-state `state_Nu`, actions, source hashes, and aggregate metrics are
   stored in `Baselines/<ra>/expert.jld2`.
7. Record selection, expert, plot, source-checkpoint, and baseline provenance
   in `experts/selection_manifest.jld2`.

`compute_unactuated_baselines.jl` complements these expert baselines with the
matching zero-action reference. For each selected Rayleigh number it evaluates
the same eight test cases in the same order for 200 control steps and writes
`Baselines/<ra>/unactuated.jld2`. The two Higher-Ra run files are loaded in
separate Julia subprocesses so their constants and state corpora cannot collide.
Existing complete results are reused only when the study, run seed, step/case
counts, run-file hash, and test-corpus hash still match.

Both Rayleigh numbers are computed sequentially by default:

```bash
julia --startup-file=no --project=. \
  Revision/Higher_Ra_Study/compute_unactuated_baselines.jl
```

The run can be restricted or intentionally repeated with, for example:

```bash
julia --startup-file=no --project=. \
  Revision/Higher_Ra_Study/compute_unactuated_baselines.jl --study ra1e5

julia --startup-file=no --project=. \
  Revision/Higher_Ra_Study/compute_unactuated_baselines.jl --study ra5e4 --overwrite
```

Default invocation:

```bash
julia --startup-file=no --project=. Revision/Higher_Ra_Study/analyze_runs.jl
```

Alternative result roots can be supplied with `--ra5e4-results` and
`--ra1e5-results`. The script can be rerun while training is still in progress;
atomic resume and best-so-far checkpoints are treated as available snapshots.
A later rerun may legitimately replace the selected expert if a better
rolling-100 candidate has appeared.

While only one result root is locally available, `--study ra5e4` or
`--study ra1e5` restricts the complete extraction, plotting, and baseline
workflow to that Rayleigh number. The default `--study all` analyzes both.

## Local layout

```text
Higher_Ra_Study/
  analyze_runs.jl
  compute_unactuated_baselines.jl
  HigherRaDistillationCorpus.jl
  prepare_corpus_workers.jl
  run_corpus_worker.jl
  execute_corpus_worker.jl
  launch_tmux_common.sh
  launch_tmux_ra5e4.sh
  launch_tmux_ra1e5.sh
  Higher_Ra_Study.md
  experts/
    selection_manifest.jld2
    ra5e4/expert.jld2
    ra1e5/expert.jld2
  Baselines/
    ra5e4/expert.jld2
    ra5e4/unactuated.jld2
    ra1e5/expert.jld2
    ra1e5/unactuated.jld2
  plots/
  Distillation_Corpuses/
    ra5e4/worker_results/
    ra1e5/worker_results/
  GO_GR_Study/
    HigherRaGOStudy.jl
    prepare_manifest.jl
    run_training_worker.jl
    analyze_configuration_worker.jl
    launch_tmux_ra5e4.sh
    launch_tmux_ra1e5.sh
    README.md
    results/
```

## Implemented second stage: separate distillation corpora

Each Rayleigh number requires its own expert-rollout distillation corpus. The
corpora never mix states, actions, or expert identities across Rayleigh
numbers. `HigherRaDistillationCorpus.jl` reuses the established compact
Package-8 corpus format while binding each worker result to its matching local
Higher-Ra expert, Higher-Ra MAT run file, and Higher-Ra initial-condition
corpus by absolute path and SHA-256.

The implementation follows the established Varying-IC distillation protocol:

- use only the selected expert for the matching Rayleigh number;
- use `varying_ic_corpus_Ra5e4.jld2` for `ra5e4` and
  `varying_ic_corpus_Ra1e5.jld2` for `ra1e5`;
- verify the Rayleigh number recorded in the state corpus and freeze the expert
  SHA-256, run-file SHA-256, state-corpus SHA-256, split, and case plan in every
  shard;
- retain the compact global `3 x 48 x 8` observations and deterministic expert
  action means;
- preserve train/validation/test isolation;
- use all 20 training bases with both mirrors and all 96 horizontal offsets:
  40 workers, 96 episodes and 19,200 samples per worker;
- use the one validation base with both mirrors and offsets 0 and 20:
  2 workers, 4 episodes and 800 samples in total;
- use both test bases with both mirrors and offsets 0 and 20:
  4 workers, 8 episodes and 1,600 samples in total;
- run every episode for 200 deterministic expert-control steps;
- store one atomic file per worker-owned shard and support restart without
  rewriting valid shards;
- keep all generated higher-Ra corpus files below
  `Higher_Ra_Study/Distillation_Corpuses/<ra>/`.

`prepare_corpus_workers.jl` validates the sources and writes only missing or
provenance-mismatching jobs. `run_corpus_worker.jl` owns one shard. The two
public launchers are deliberately separate, while
`launch_tmux_common.sh` contains their shared restart-safe tmux mechanics.
If more jobs exist than `--max-workers`, a persistent tmux slot processes its
assigned shards sequentially.

### Launch commands

For `Ra = 1e5`:

```bash
bash Revision/Higher_Ra_Study/launch_tmux_ra1e5.sh --split train
bash Revision/Higher_Ra_Study/launch_tmux_ra1e5.sh --split validation --max-workers 2
bash Revision/Higher_Ra_Study/launch_tmux_ra1e5.sh --split test --max-workers 4
```

For `Ra = 5e4`:

```bash
bash Revision/Higher_Ra_Study/launch_tmux_ra5e4.sh --split train
bash Revision/Higher_Ra_Study/launch_tmux_ra5e4.sh --split validation --max-workers 2
bash Revision/Higher_Ra_Study/launch_tmux_ra5e4.sh --split test --max-workers 4
```

`--split all` prepares all 46 shards for one Rayleigh number. The default
concurrency is 40, so the smaller validation/test tail is queued behind the
first slots. `--preview` performs source and corpus-plan validation and prints
the commands without starting tmux. Valid completed shards are skipped;
`--overwrite` explicitly requests regeneration.

The launchers default to `experts/<ra>/expert.jld2`. Consequently the `ra1e5`
launcher is ready for the expert already published by `analyze_runs.jl`. The
`ra5e4` launcher intentionally fails before launch until the corresponding
local expert exists.

## Implemented third stage: GO and GR apprentice studies

`GO_GR_Study/` contains the paired GO/GR study modeled on Package 8. It
contains:

- a shared study module defining both Rayleigh numbers, GO/GR configurations,
  seeds, regularization strengths, checkpoint schedules, and atomic paths;
- manifest preparation that verifies the local expert and matching local
  distillation-corpus identities;
- one restart-safe training worker per Rayleigh number, method, strength, and
  replicate;
- two independent tmux launchers with configuration and strength filters;
- validation-only Pareto construction over active sensor groups/inputs and
  autoregressive validation MSE;
- frozen candidate selection before any terminal test-set rollout;
- deterministic closed-loop test evaluation on the matching eight higher-Ra
  test cases;
- Package-8-style aggregate metrics, Pareto plots, test plots, and provenance
  manifests.

GO and GR are compared under paired initialization, corpus batches, and
evaluation cases. Noise robustness is not part of this initial higher-Ra study
and should only be added as a separately specified post-hoc study.

The implemented grid contains only `go-gc`, `go-sc`, `gr-gc`, and `gr-sc`.
Each configuration uses its five frozen strengths and three paired replicates.
Mask thresholds are `(0.0, 0.003, 0.006, 0.012)`. Each configuration analyzer
selects the sparsest pooled Pareto point independently under validation-MSE
quality thresholds `0.03`, `0.015`, and `0.0075`, deduplicates identical
selections, freezes all selected candidates, and then evaluates up to three of
them on the matching eight-case Higher-Ra test set.

The two independent launchers are:

```bash
bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra5e4.sh
bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra1e5.sh
```

Each full launcher starts 60 training workers and four analysis/wait workers.
Detailed restart, filter, selection, and output documentation is in
`GO_GR_Study/README.md`.

## Implemented fourth stage: paper figures and tables

`GO_GR_Study/make_paper_figures.jl` produces separate publication artifacts
for `ra5e4` and `ra1e5`. Per Rayleigh number it creates one four-panel Pareto
figure and one four-panel sensor-mask figure covering `go-gc`, `go-sc`,
`gr-gc`, and `gr-sc`, plus a Markdown/CSV table of every distinct tested
quality-threshold candidate. The Pareto panels show the three validation-MSE
thresholds `0.03/0.015/0.0075` and their selected candidates. Table rows list
all thresholds represented by the candidate and omit thresholds without a
qualifying point; Expert and Unactuated remain the first and final rows.

For the sensor-mask figure, test-near-expert is defined before inspecting the
candidate values as at most 5% higher mean test-set `state_Nu` than the expert.
Since lower `state_Nu` is better, candidates better than the expert also
qualify. The fewest-input candidate in this band is used, followed by active
groups, test mean, validation MSE, and candidate ID as deterministic
tie-breakers. The tolerance is an explicit command-line option, and the script
refuses to substitute a candidate outside it.

## Remaining work

- Interpret the pooled fronts and terminal-test trade-offs for both Rayleigh
  numbers after all analysis/test workers have completed.
