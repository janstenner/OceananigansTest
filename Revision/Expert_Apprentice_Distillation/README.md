# Expert–Apprentice Distillation

This directory contains the shared offline distillation infrastructure used by
revision Packages 6, 7, and 8.

## Files

- `Expert_Apprentice.jl`: revision copy of the existing MAT
  expert–apprentice implementation and the starting point for the shared
  Package-6/7/8 training runner.
- `DistillationCorpus.jl`: directly includable worker generation, atomic JLD2
  persistence, expert discovery, corpus loading, and global-to-local
  observation reconstruction.
- `run_corpus_worker.jl`: one Fixed-IC or Varying-IC worker entry point.
- `execute_corpus_worker.jl`: post-run-file adapter kept separate for Julia
  world-age-safe access to the selected MAT environment.
- `prepare_corpus_workers.jl`: deterministic worker manifest creation.
- `launch_tmux.sh`: detached, restart-safe parallel launch.
- `test_distillation_corpus.jl`: small callback-based storage and merge tests.

## Corpus layout

A Varying-IC worker owns one `(split, base_seed, mirror)` combination and
normally evaluates offsets `0:95`, each for 200 deterministic expert control
steps. The default training launch therefore contains 40 workers:

```text
20 training bases × 2 mirror choices = 40 workers
```

Validation has two workers and test has four workers. Fixed IC uses one worker
and one 200-step episode; the loader exposes that same dataset object as its
training, validation, and test set.

Workers store the unique global `3×48×8` sensor tensor and the `1×12` expert
action means as `Float32`. The overlapping `360×12` MAT observation is rebuilt
losslessly through `local_mat_observation`; `distillation_batch` reconstructs
only selected mini-batches and never expands the complete corpus in memory.
This reduces the complete Varying
training corpus from about 12.4 GiB to about 3.33 GiB.

Every worker writes one descriptive JLD2 atomically below `worker_results/`.
Including `DistillationCorpus.jl` loads every available complete worker file
and merges it by protocol and split into `DISTILLATION_CORPUS`. Corpus workers
set `DISTILLATION_SKIP_AUTOLOAD=true` to avoid loading the growing training set
while generating another shard.
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

Generate the Fixed-IC dataset:

```bash
Revision/Expert_Apprentice_Distillation/launch_tmux.sh \
  --protocol fixed --fixed-expert-path /path/to/agent.jld2
```

Completed matching files are skipped. Use `--overwrite` only when a shard is
intentionally regenerated. Jobs run in detached tmux sessions, use a systemd
sleep inhibitor by default, and write a persistent manifest and logs below
`worker_results/launches/`.
