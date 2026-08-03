# MAT stability experiment

This directory runs the paired MAT ablation from revision package 3. Each
worker handles one protocol and one replicate, then trains the three
configurations sequentially with the same seed plan:

| Configuration | Separate value chain | LayerNorm | Self-attention first | Mean-token feedback |
|---|---:|---:|---:|---:|
| `python_like` | false | true | true | false |
| `modified_half` | true | false | false | false |
| `modified_full` | true | false | false | true |

Dropout remains `0.0` in the underlying run files and is not an experimental
factor.

## Files

- `MATStabilityExperiment.jl` contains the seed plan, exact-episode training,
  pairing checks, locking, and atomic JLD2 persistence.
- `run_worker.jl` is the command-line entry point for one protocol/replicate.
- `launch_tmux.sh` starts all ten detached server workers.
- `collect_results.jl` validates the 30 expected result files and creates
  metrics and plots.

## Preflight

Run one zero-episode worker before submitting the experiment:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_Stability/run_worker.jl \
  --protocol fixed --replicate 1 --dry-run

julia --startup-file=no --project=. \
  Revision/MAT_Stability/run_worker.jl \
  --protocol varying --replicate 1 --dry-run
```

These runs construct and save all three agents, but perform no training. The
worker verifies that:

- all configurations use the same policy RNG state and initial-condition
  probe;
- parameters shared by all architectures are byte-identical;
- the main and separate value chains start with equal values but independent
  arrays;
- `modified_half` and `modified_full` have byte-identical complete networks.

Dry-run artifacts are kept below `results/dry_run` and do not collide with
production results.

## Server launch

Preview all planned sessions:

```bash
bash Revision/MAT_Stability/launch_tmux.sh --preview
```

Start only the five Fixed-IC workers:

```bash
bash Revision/MAT_Stability/launch_tmux.sh --protocol fixed
```

Start only the five Varying-IC workers:

```bash
bash Revision/MAT_Stability/launch_tmux.sh --protocol varying
```

Start both protocols, for ten workers in total:

```bash
bash Revision/MAT_Stability/launch_tmux.sh
```

Each worker runs its three configurations in sequence. Fixed IC uses exactly
2,000 episodes per configuration; Varying IC uses exactly 4,000. The sessions
are detached and therefore continue after the SSH connection is closed.
Each Julia worker is started through `systemd-inhibit` and holds a blocking
`sleep:idle:shutdown` inhibitor for exactly its process lifetime. Inspect the
active locks with `systemd-inhibit --list`. For local systems without systemd,
`--no-systemd-inhibit` disables the wrapper explicitly.

Useful commands:

```bash
tmux ls
tmux attach -t mat_fixed_01
tail -f Revision/MAT_Stability/results/logs/mat_fixed_01.log
```

Set `JULIA_BIN=/path/to/julia` if `julia` is not on `PATH`. Set
`MAT_STABILITY_RESULTS_DIR=/path/to/results` to store the result tree
elsewhere. `--dry-run-workers` launches zero-episode workers for the selected
protocols; `--overwrite` explicitly replaces matching completed results.
`SYSTEMD_INHIBIT_WHAT` may override the default inhibited operations.

Workers create a lock per protocol/replicate. A restart skips complete,
matching result files and resumes at the first missing or failed
configuration. Remove `.worker.lock` only after confirming that no
corresponding process or tmux session is active.

## Results

Production files follow this layout:

```text
results/
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

Every configuration is saved atomically immediately after training. The file
contains the agent, reward histories, errored episodes, runtime, configuration,
seed plan, parameter counts, initial parameter hashes, IC probe, Varying-IC
selection trace, relevant run parameters, source-file hashes, Julia version,
host, timestamps, and Git states. A failed configuration is recorded with its
error and partial reward history.

The collector can be run at any time:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_Stability/collect_results.jl
```

Missing protocols, replicates, and configurations are ignored. The collector
processes every complete, valid result file that is currently present and
rechecks pairing wherever multiple configurations of one replicate are
available. It writes `metrics.csv`, `summary.jld2`, and the currently possible
learning-curve, final-performance, and runtime plots below
`results/collected`.
