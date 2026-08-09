# Package 6 — GO Sensitivity

This directory owns the Package-6 experiment orchestration. Shared apprentice
training, distillation-corpus loading, mask generation, and Pareto storage stay
in `Revision/Expert_Apprentice_Distillation`.

## One-seed strength calibration

`run_strength_calibration_pilot.jl` performs the technical calibration that
precedes the five-strength Package-6 production sweep. The completed baseline
is retained unchanged:

| Protocol | Grouping | Strengths |
|---|---|---|
| Fixed IC | grouped channels | `0.01`, `0.03`, `0.09` |
| Fixed IC | separate channels | `0.01`, `0.02`, `0.03` |
| Varying IC | grouped channels | `0.003`, `0.008`, `0.025` |
| Varying IC | separate channels | `0.003`, `0.008`, `0.025` |

Those twelve baseline runs used apprentice seed `600601`, 6,000 Fixed or
10,000 Varying updates, and the historical Fixed regression learning rate
`1e-4` (Varying already used `2e-4`). They are not resumed, extended, or
recomputed.

The runner now adds the following extension runs:

| Protocol | Grouping | Additional strengths | Updates |
|---|---|---|---:|
| Fixed IC | grouped channels | `0.003`, `0.006`, `0.06` | 9,000 |
| Fixed IC | separate channels | `0.0015`, `0.003`, `0.006` | 9,000 |
| Varying IC | grouped channels | `0.04`, `0.06` | 15,000 |
| Varying IC | separate channels | `0.04`, `0.06` | 15,000 |

All ten extension runs use regression learning rate `2e-4`, apprentice seed
`600601`, and 1.5 times the previous update budget. Their run IDs explicitly
contain phase, learning rate, and budget, so they coexist with the baseline
without collisions. All calibration runs evaluate native GO sparsity only,
with autoregressive validation every 25 updates starting at update 0.
Hard-threshold candidates and teacher-forced diagnostics are omitted because
this pilot only calibrates the strength scale. The training runner checks
test-corpus coverage and expert provenance but never evaluates test samples
itself.

Because learning rate and budget changed together with the added strengths,
baseline-versus-extension comparisons are exploratory rather than a controlled
one-factor strength comparison. All future Package-6/7/8 apprentice runs use
the extension settings (`2e-4`, 9,000 Fixed updates, 15,000 Varying updates).

Each protocol/grouping combination is executed sequentially in a fresh Julia
process. Within that process, all additional strengths start from deep copies
of the same apprentice and use the same batch-order RNG. Fresh processes are
required because protocol and channel grouping are fixed when the shared
apprentice file is included. Sequential execution also avoids holding multiple
copies of the multi-gigabyte Varying corpus in memory.

From Windows PowerShell in the project root:

```powershell
julia --project=. .\Revision\GO_Sensitivity\run_strength_calibration_pilot.jl
```

Or from an activated Julia REPL:

```julia
include("Revision/GO_Sensitivity/run_strength_calibration_pilot.jl")
run_strength_calibration_pilot()
```

The runner resumes interrupted strength runs, skips complete ones, and writes
results below `Revision/GO_Sensitivity/results/strength_calibration/`. A command
preview that starts no child process is available as
`run_strength_calibration_pilot(; preview = true)`.

After all ten extension runs finish, run the inspector with:

```powershell
julia --project=. .\Revision\GO_Sensitivity\inspect_strength_calibration_pilot.jl
```

The inspector combines the complete baseline and extension blocks and creates
one validation SVG for each Fixed/Varying and grouped/separate-channel
combination below `results/strength_calibration/analysis/`. Incomplete
extension blocks are ignored, so a running extension never yields a mixed
partial plot. Every validation checkpoint is an unconnected circle, colored
by regularization strength. Larger diamonds with a black outline mark the pooled nondominated
checkpoints while retaining the corresponding strength color. The y-axis uses
a logarithmic scale because the calibration spans multiple orders of magnitude
in validation MSE.

By default, and only after all four extension blocks are complete, the
inspector starts one fresh Julia process per combination and evaluates **every
candidate retained by all baseline and extension Pareto archives** in closed
loop together with the MAT expert. Fixed IC uses the one shared 200-step
episode. Varying IC uses the eight predeclared test episodes
(two bases, original and mirrored, offsets 0 and 20). Runs are cached by expert,
candidate, and test case, so an interruption can be resumed without repeating
complete episodes.

Each combination produces:

- a reward-curve SVG with the expert in black and apprentice mean curves
  colored by regularization strength;
- a cumulative 200-step-return box plot with all episode points and the mean;
- a per-case return CSV and an aggregate JLD2 result.

These files are stored below
`results/strength_calibration/analysis/calibration_test_diagnostic/`. For Fixed
IC, the box plot has only one observation per controller and is therefore a
compact comparison rather than a distribution; for Varying IC it summarizes
the eight test episodes in the same style as `rIC-validation.jl`.

To regenerate only the four validation/Pareto plots, without running any
closed-loop episode or touching the Varying test split, use:

```julia
include("Revision/GO_Sensitivity/inspect_strength_calibration_pilot.jl")
inspect_strength_calibration_pilot(closed_loop_test = false)
```

### Test-split warning

The default closed-loop diagnostic consumes and exposes the Varying test split.
Its result files are marked `scientific_selection_allowed = false`; they must
not be used to choose strengths or candidates. Once these curves have been
viewed, the current eight episodes are no longer an untouched final Package-8
test set. A later final held-out claim therefore requires newly generated and
frozen test bases (and matching expert rollout shards), or the present split
must be reported explicitly as an exploratory diagnostic split.

## Fixed-IC technical pilot

`run_fixed_pilot.jl` runs one explicitly non-scientific local pilot:

- Fixed IC
- GO with `regularization_strength = 0.09`
- apprentice seed `600601`
- 6,000 optimizer/proximal updates
- batch size 20
- candidate evaluation every 5 updates starting at update 2,000
- autoregressive validation on the shared 200-sample Fixed corpus
- native sparsity plus two `:pilot_only` hard-threshold checks
- no post-pruning finetuning

It requires these existing files:

```text
Revision/Expert_Apprentice_Distillation/experts/fixed/agent.jld2
Revision/Expert_Apprentice_Distillation/worker_results/fixed/fixed_shared.jld2
```

The runner is plain Julia. It does not use Bash, tmux, or server-specific
process handling. From Windows PowerShell in the project root:

```powershell
julia --project=. .\Revision\GO_Sensitivity\run_fixed_pilot.jl
```

Alternatively, from a fresh Julia REPL started with the project activated:

```julia
include("Revision/GO_Sensitivity/run_fixed_pilot.jl")
run_fixed_pilot()
```

Results are written below `Revision/GO_Sensitivity/results/fixed_pilot/` and
ignored by Git. An interrupted run resumes from `resume/latest.jld2`. A
completed run is reported and not repeated. A numerical failure is retained
for inspection and is not silently resumed.

## Inspecting the pilot

`inspect_fixed_pilot.jl` first reads the lightweight configuration, evaluation,
archive, and summary files. By default it then executes one deterministic
200-step Fixed-IC closed-loop episode for the best-matching retained native
apprentice and compares it with the MAT expert. From Windows PowerShell in the
project root:

```powershell
julia --project=. .\Revision\GO_Sensitivity\inspect_fixed_pilot.jl
```

Or from an activated Julia REPL:

```julia
include("Revision/GO_Sensitivity/inspect_fixed_pilot.jl")
inspection = inspect_fixed_pilot()
```

With no argument, the newest completed Fixed-IC pilot is selected. The
inspector prints the complete native-GO trajectory, the final threshold
comparison, and the retained Pareto archive. It also writes `overview.txt`,
`candidate_history.csv`, `historical_pareto_front.csv`, and
`current_archive_front.csv` into the selected run's `analysis` directory. A
PlotlyJS figure named `validation_pareto_checkpoints.svg` places every
validation checkpoint by active groups and autoregressive validation MSE.
Checkpoints are shown as unconnected points. Hover data also reports update,
active global inputs, and active global sensor locations.

The first closed-loop inspection loads the expert and selected apprentice and
runs the real Oceananigans environment twice from the same Fixed initial
condition. It stores the actual mean environment reward, the full-state global
Nusselt number, and all twelve actions for every control step. The expert
episode is cached below `results/fixed_pilot/closed_loop_cache` using the expert
and initial-condition SHA-256 identities. The apprentice episode is cached in
the run's `analysis` directory using its Pareto candidate ID. Later inspections
reuse matching JLD2 caches.

Closed-loop outputs are `closed_loop_summary.txt`,
`closed_loop_reward_comparison.csv`, `closed_loop_reward_comparison.svg`, and
the candidate-specific apprentice episode JLD2. The plotted reward is the mean
of the twelve environment rewards at each control step; higher is better. The
full-state global Nusselt number is included in the hover data and CSV.

To skip all model loading and closed-loop work:

```julia
inspection = inspect_fixed_pilot(; closed_loop = false)
```

If a native run front contains multiple candidates, the default is the one
with the lowest autoregressive validation MSE. A specific retained native
candidate can instead be selected with
`closed_loop_candidate_id = "<candidate-id>"`.

The pilot validates the real end-to-end path and measures local runtime. Its
thresholds and results are not Package-6 production decisions. Production
training budgets, evaluation intervals, apprentice seeds, and the complete
Package-7/8 threshold list are frozen only after inspecting this pilot.
