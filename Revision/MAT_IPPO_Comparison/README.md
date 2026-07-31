# Package 4: MAT versus IPPO

This directory runs `modified_full` MAT against parameter-sharing IPPO from
`Revision/Run_Files` under Fixed IC and Varying IC. Fixed runs use 2,000
episodes and Varying runs use 4,000 episodes.

## Recommended workflow

Start the five new seed pairs first:

```bash
Revision/MAT_IPPO_Comparison/launch_tmux.sh \
  --n-runs 5 --look-for-imports false --max-workers 20
```

This creates five seed pairs and starts 20 jobs: MAT and IPPO for both
protocols. The plan reserves no run numbers for Package 3; a stable ID is
derived from `(run_seed, ic_seed)`.

When Package 3 is complete, inspect and perform the import:

```bash
julia --project=. Revision/MAT_IPPO_Comparison/import_package3.jl \
  --n-runs 5 --preview

julia --project=. Revision/MAT_IPPO_Comparison/import_package3.jl \
  --n-runs 5
```

The importer accepts only complete `modified_full` fixed/varying pairs with
matching seeds and budgets. It records source paths and SHA-256 hashes and
adopts the imported Varying-IC trace as authoritative. If fewer than five pairs
exist, it fails without generating replacement seeds.

Then start the missing IPPO partners:

```bash
Revision/MAT_IPPO_Comparison/launch_tmux.sh \
  --n-runs 5 --look-for-imports true --max-workers 20
```

For five imports, this creates exactly ten IPPO workers. Each IPPO worker also
validates its imported MAT partner, avoiding separate MAT validation workers.

Use `--protocol fixed` or `--protocol varying` to run one protocol only. A
later invocation for the other protocol resumes the same unfinished seed pairs.

## Launcher behavior

The launcher invokes Julia for planning and again in each generated worker-slot
script. Every invocation uses `--project=$PROJECT_ROOT`, so the repository
project is activated automatically. Set `JULIA_BIN` if Julia is not on `PATH`.

`--preview` prints the proposed plan and exact worker commands but persists no
plan, copies no imports, and starts no tmux sessions.

`--max-workers N` defaults to 20. At most that many detached tmux slots process
all jobs in queues. MAT and IPPO for a seed pair are adjacent in the manifest
and occupy different slots when capacity permits. Sessions survive SSH
disconnects and deliberately remain open after finishing. Their logs and job
manifest are under `results/launches/`.

`--overwrite` reruns the most recent `--n-runs` entries of the selected origin.
Ordinary restarts skip compatible complete files and resume only missing work.

## Reproducibility and storage

`MATIPPOExperiment.jl` owns the atomically written and locked
`results/run_plan.jld2`. It stores each `run_seed`, `ic_seed`, origin, and exact
4,000-case Varying-IC schedule `(split, base_seed, mirror, offset)`. Workers
consume this plan and never choose their own seeds or cases.

Every launcher call is also assigned a batch ID. If a batch is only partly
complete, repeating the original command resumes that entire batch and never
fills its missing jobs with newly generated seed pairs. A different
`--n-runs` value is rejected until the unfinished batch is resumed with its
original size.

Training checkpoints are stored at:

```text
results/runs/<run_id>/<fixed|varying>/<mat|ippo>.jld2
```

Validation sidecars use `<algorithm>_validation.jld2`. Separate atomic files
ensure that a validation failure cannot destroy a complete end checkpoint.
Failed training and validation files retain their error details.

Fixed validation uses one deterministic mean-action rollout on the fixed IC.
Varying validation uses the sole validation base in all four combinations of
`mirror=false/true` and `offset=0/20`. MAT expert candidates are ranked only by
these validation results; test bases are not used.

## Collection

Run at any time:

```bash
julia --project=. Revision/MAT_IPPO_Comparison/collect_results.jl
```

The collector uses every valid complete result currently present. It assumes
neither ten runs nor complete protocols/algorithm pairs and combines imported
and new results. It regenerates known plots to remove stale output.

`results/analysis/` contains PNG/PDF learning curves (individual runs,
rolling-50 median with 25/75-percentile ribbon, and dashed mean), individual
curve plots, final-last-100 performance, paired differences, validation and
runtime plots. CSV files contain run diagnostics, pairing, failures, MAT expert
ranking, and curve statistics including mean, standard deviation, median,
quartiles, and the actual mean deviation below and above the mean. A compact
`collected_results.jld2` stores the same analysis without duplicating agents.
