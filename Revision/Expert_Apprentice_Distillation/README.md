# Expert–Apprentice Distillation

This directory contains the shared offline distillation infrastructure used by
revision Packages 6, 7, and 8.

## Files

- `Expert_Apprentice.jl`: shared MAT apprentice training, real corpus
  validation, native/group-consistent hard-threshold mask generation, and the
  canonical `:go`, `:gr`, `:group_lasso`, and `:growl` method configurations.
- `DistillationCorpus.jl`: directly includable worker generation, atomic JLD2
  persistence, expert discovery, corpus loading, and global-to-local
  observation reconstruction.
- `ParetoArchive.jl`: shared per-run candidate archive, strict two-objective
  dominance, atomic manifests, reference-based model garbage collection, and
  independent resume checkpoints.
- `run_corpus_worker.jl`: one Fixed-IC or Varying-IC worker entry point.
- `execute_corpus_worker.jl`: post-run-file adapter kept separate for Julia
  world-age-safe access to the selected MAT environment.
- `prepare_corpus_workers.jl`: deterministic worker manifest creation.
- `launch_tmux.sh`: detached, restart-safe parallel launch.
- `test_distillation_corpus.jl`: small callback-based storage and merge tests.
- `test_pareto_archive.jl`: archive dominance, shared-checkpoint, garbage
  collection, rebuild, and resume tests.

## Apprentice training and candidates

Experiment scripts own the scientific policy. They construct
`ApprenticeTrainingConfig`, `CandidateSchedule`, the complete vector of
`HardThresholdSpec`s, and a `ParetoArchiveManager`, then call
`train_apprentice!`. One training update is exactly one optimizer update;
`proximal_interval` controls proximal applications independently. Proximal
pruning keeps at least `minimum_active_groups` groups (default `1`) by randomly
restoring groups pruned in the current step with the resumable training RNG.

Validation always uses the loaded corpus `:validation` split. Autoregressive
action matching is the default Pareto objective; teacher-forced matching can
be stored as a diagnostic. Native zeros and post-hoc hard-threshold masks are
separate candidate types and separate Pareto scopes. A hard-threshold
candidate therefore cannot delete a Package-6 native-front checkpoint.
Thresholding is group-consistent and never mutates the model.

Per-update evaluation JLD2s are the atomic training-time event log. Experiments
may opt into slim records that keep the scientific objectives and group mask
while moving expanded masks and group importances into the retained Pareto
model checkpoint. Every survivor from one training update shares that update's
single model checkpoint. Completed experiments may atomically consolidate the
verified shards into one `evaluations.jld2`.
`resume/latest.jld2` is separate from the scientific archive. Logical Pareto
pruning happens after every evaluation; unreferenced model files are removed
periodically and on finalization.

## Corpus layout

A Varying-IC worker owns one `(split, base_seed, mirror)` combination. Training
workers evaluate offsets `0:95`; validation and test workers evaluate only the
fixed offsets `0` and `20`. Every episode contains 200 deterministic expert
control steps. The default training launch therefore contains 40 workers:

```text
20 training bases × 2 mirror choices = 40 workers
```

Validation has two workers with four episodes and 800 samples in total. Test
has four workers with eight episodes and 1,600 samples in total. Fixed IC uses
one worker and one 200-step episode; the loader exposes that same dataset
object as its training, validation, and test set.

Workers store the unique global `3×48×8` sensor tensor and the `1×12` expert
action means as `Float32`. The overlapping `360×12` MAT observation is rebuilt
losslessly through `local_mat_observation`; `distillation_batch` reconstructs
only selected mini-batches and never expands the complete corpus in memory.
This reduces the complete Varying
training corpus from about 12.4 GiB to about 3.33 GiB.

Every worker writes one descriptive JLD2 atomically below `worker_results/`.
Including `DistillationCorpus.jl` loads every available complete worker file
and merges it by protocol and split into `DISTILLATION_CORPUS`. Setting
`DISTILLATION_AUTOLOAD_PROTOCOL` to `fixed` or `varying` before inclusion loads
only the selected protocol; the default is `all`. Experiment runners should
select their protocol so a Fixed-only process does not load the multi-gigabyte
Varying corpus. Corpus workers set `DISTILLATION_SKIP_AUTOLOAD=true` to avoid
loading the growing training set while generating another shard.
Package runners call `assert_distillation_coverage` before production training;
partial available shards remain loadable for inspection and development.

## Expert checkpoints

Production workers require an expert checkpoint. Resolution order is:

1. `--expert-path`
2. `DISTILLATION_FIXED_EXPERT_PATH` or
   `DISTILLATION_VARYING_EXPERT_PATH`
3. `experts/fixed/agent.jld2` or `experts/varying/agent.jld2`
4. the corresponding default Revision Run-File save path

The checkpoint must contain `agent` or `expert`. Its SHA-256 identifier is
stored in every worker file, and merging rejects shards from different
experts. `--allow-fresh-expert` is an explicit smoke-test-only escape hatch;
there is no silent fresh-agent fallback.

The tracked production checkpoints use the final hybrid MAT expert selection:
Fixed is the unchanged Package-4 validation-rank-1 MAT agent
`seed_dfe17c7e95fcbb6d`, while Varying is the threshold winner
`seed_5954219d6cc69d5c` after 703 additional episodes. Both files contain only
the agent and an empty trajectory with capacity-one backing buffers. Selection
details and deterministic full-state Nusselt evaluations are recorded in
`../MAT_expert_training/results/results_notes.md`.

## Launch

Preview the 40 Varying-IC training workers:

```bash
Revision/Expert_Apprentice_Distillation/launch_tmux.sh \
  --protocol varying --split train --preview
```

Start them after setting the final expert:

```bash
Revision/Expert_Apprentice_Distillation/launch_tmux.sh \
  --protocol varying --split train \
  --varying-expert-path /path/to/agent.jld2
```

After training shards exist, generate the two Varying validation workers and
the four Varying test workers separately. This avoids needlessly checking all
40 large training shards while preparing either launch:

```bash
bash Revision/Expert_Apprentice_Distillation/launch_tmux.sh \
  --protocol varying --split validation --max-workers 2 \
  --varying-expert-path Revision/Expert_Apprentice_Distillation/experts/varying/agent.jld2

bash Revision/Expert_Apprentice_Distillation/launch_tmux.sh \
  --protocol varying --split test --max-workers 4 \
  --varying-expert-path Revision/Expert_Apprentice_Distillation/experts/varying/agent.jld2
```

Completed matching shards are skipped, so neither command needs `--overwrite`
for a restart.

Generate the Fixed-IC dataset:

```bash
Revision/Expert_Apprentice_Distillation/launch_tmux.sh \
  --protocol fixed --fixed-expert-path /path/to/agent.jld2
```

Completed matching files are skipped. Use `--overwrite` only when a shard is
intentionally regenerated. Jobs run in detached tmux sessions and write a
persistent manifest and logs below `worker_results/launches/`.
