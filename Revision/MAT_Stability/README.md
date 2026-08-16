# MAT stability experiment

This directory runs the paired MAT ablation from revision package 3. The
launcher creates one independent worker for every protocol, replicate, and MAT
configuration: 15 Fixed-IC and 15 Varying-IC workers in a complete launch.

| Configuration | Separate value chain | LayerNorm | Self-attention first | Mean-token feedback |
|---|---:|---:|---:|---:|
| `python_like` | false | true | true | false |
| `modified_half` | true | false | false | false |
| `modified_full` | true | false | false | true |

Dropout remains `0.0` in the underlying run files and is not an experimental
factor.

## Frozen experiment plan

`prepare_runs.jl` runs synchronously before tmux is touched. It atomically
creates `results/run_plan.jld2` and a launch-specific `jobs.tsv`. The immutable
plan contains the five run/IC seed pairs and the complete 4,000-episode
Varying-IC choice sequence for every replicate. Thus three separately running
configuration workers load the same initialization seed and exactly the same
ordered `(base_seed, mirror, offset)` choices.

The plan is checked against the deterministic seed generator, corpus, three
configuration names, replicate set, and episode budgets whenever a launch is
prepared. A worker refuses to run without that persisted plan. Existing
complete result files with matching seeds, budgets, configuration, and current
source hashes are omitted from the manifest unless `--overwrite` is supplied.
This means results produced before a run-file or MAT implementation change are
scheduled again automatically.

## Files

- `MATStabilityExperiment.jl` contains plan generation, exact-episode training,
  pairing metadata, per-result locking, and atomic JLD2 persistence.
- `prepare_runs.jl` freezes/verifies the plan and writes the worker manifest.
- `run_worker.jl` runs one protocol/replicate/configuration job.
- `launch_tmux.sh` starts up to 30 independent detached workers.
- `collect_results.jl` validates available results, including all paired-design
  invariants, and creates metrics and plots.

## Preflight

Preview the complete launch without writing the plan or starting tmux:

```bash
bash Revision/MAT_Stability/launch_tmux.sh --preview
```

Run all zero-episode construction checks in independent sessions:

```bash
bash Revision/MAT_Stability/launch_tmux.sh --dry-run-workers
```

A single worker can also be run directly after `prepare_runs.jl` has created a
plan:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_Stability/run_worker.jl \
  --protocol fixed --replicate 1 --config python_like --dry-run
```

The collector rechecks that configurations in a replicate have identical
shared initial parameters, policy RNG probes, IC probes, and Varying-IC
sequences. It also checks that `modified_half` and `modified_full` start with
byte-identical complete networks and that separate value-chain arrays equal
the main chain initially without sharing storage.

## Server launch

Start all 30 workers:

```bash
bash Revision/MAT_Stability/launch_tmux.sh
```

Start only the 15 workers of one protocol:

```bash
bash Revision/MAT_Stability/launch_tmux.sh --protocol fixed
bash Revision/MAT_Stability/launch_tmux.sh --protocol varying
```

Fixed IC uses exactly 2,000 episodes per configuration; Varying IC uses exactly
4,000. Every tmux session executes only one configuration and closes when that
job finishes. Sessions are detached and survive an SSH disconnect.

Useful commands (the exact log directory is printed by the launcher):

```bash
tmux ls
tmux attach -t mat_fixed_01_python_like
tail -f Revision/MAT_Stability/results/launches/<launch-id>/fixed_01_python_like.log
```

Set `JULIA_BIN=/path/to/julia` if needed. Either set
`MAT_STABILITY_RESULTS_DIR=/path/to/results` or pass `--results-dir PATH` to
relocate the result tree. `--overwrite` schedules matching completed jobs
again.

Each result has its own `<result>.lock` directory, so unrelated configurations
can run concurrently. Remove such a lock only after confirming that no
corresponding process or tmux session is active.

## Results

Production files retain the original collector-compatible layout:

```text
results/
├── run_plan.jld2
├── launches/
│   └── <launch-id>/
│       ├── jobs.tsv
│       └── fixed_01_python_like.log
├── fixed/
│   └── replicate_01/
│       ├── python_like.jld2
│       ├── modified_half.jld2
│       └── modified_full.jld2
└── varying/
    └── replicate_01/
        ├── python_like.jld2
        ├── modified_half.jld2
        └── modified_full.jld2
```

Every configuration is saved atomically immediately after training. Failed
jobs retain their error and partial reward history and are eligible for a
later relaunch. Collect all currently complete results with:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_Stability/collect_results.jl
```

Missing results are normal. The collector writes `metrics.csv`, `summary.jld2`,
and the currently possible plots below `results/collected`.
