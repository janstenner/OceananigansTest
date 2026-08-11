# Package 6 — GO Sensitivity

This directory owns the Package-6 experiment orchestration. Shared apprentice
training, distillation-corpus loading, mask generation, and Pareto storage stay
in `Revision/Expert_Apprentice_Distillation`.

## Package-6 production study

The production study is SC-only and contains 30 GO runs plus six paired GR
references. Fixed and Varying both use GO strengths `0.0015`, `0.003`,
`0.006`, `0.01`, and `0.03`; GR uses `0.00004` and `0.0001`, respectively.
Three new apprentice/batch seed pairs come from `StableRNG(20260810)`. Fixed
runs use 35,000 updates, Varying runs 50,000, regression learning rate `2e-4`,
and validation every 25 updates from update 0. Package 6 uses native sparsity
only: no thresholding, fine-tuning, or result-dependent stopping.

Preview the 36 training and two analysis sessions from the project root:

```bash
bash Revision/GO_Sensitivity/launch_study_tmux.sh --preview
```

Start the complete study:

```bash
bash Revision/GO_Sensitivity/launch_study_tmux.sh
```

Useful restricted launches are:

```bash
bash Revision/GO_Sensitivity/launch_study_tmux.sh --protocol fixed
bash Revision/GO_Sensitivity/launch_study_tmux.sh --protocol varying
bash Revision/GO_Sensitivity/launch_study_tmux.sh --analysis-only
bash Revision/GO_Sensitivity/launch_study_tmux.sh --analysis-only --parallel-test
bash Revision/GO_Sensitivity/launch_study_tmux.sh --results-dir /path/to/results/study
```

`systemd-inhibit` is enabled by default, matching the other server studies.
Use `--no-systemd-inhibit` only when the server environment intentionally does
not need or provide it. Every tmux session closes when its process exits.

Short results live at paths such as
`results/study/fixed/go/s01/r01` and `results/study/varying/gr/r03`.
Each launch writes logs, `jobs.tsv`, launch metadata, and a JLD2 study manifest
below `results/study/launches/<launch-id>/`. Complete runs are identity-checked
and skipped; interrupted runs resume. Failed runs remain explicitly failed and
require `--retry-failed` after their cause has been repaired.

The Fixed and Varying analysis workers start concurrently with training and
wait for their 18 runs. They audit configurations, pairings, corpus/expert
identity, hashes, budgets, and evaluation coverage; compute the complete
offline stability metric set; write compact SVGs, an interactive 3D HTML plot,
CSV/JLD2 metrics, and an English `report.md`; then freeze the validation-only
GO candidate manifest before launching any terminal test episode. Fixed uses
one shared 200-step test episode and Varying all eight existing test episodes.
GR is an offline reference and is never test-selected.

`--parallel-test` gives every controller/case terminal test episode its own
Julia process. With two frozen GO candidates this launches three Fixed and 24
Varying episode processes (27 total when both protocol analyses reach the test
stage). Episode-specific status files and logs make this mode resumable;
already complete valid caches are reused. It trades higher peak memory use for
lower wall-clock time.

Run the implementation tests locally with:

```powershell
julia --startup-file=no --project=. .\Revision\GO_Sensitivity\test_package6_study.jl
julia --startup-file=no --project=. .\Revision\GO_Sensitivity\test_package6_wait_worker.jl
julia --startup-file=no --project=. .\Revision\GO_Sensitivity\test_package6_plot_smoke.jl
julia --startup-file=no --project=. .\Revision\GO_Sensitivity\test_package6_test_worker_world_age.jl
```

The technical calibration below is retained as internal provenance. It is not
part of the Package-6 scientific result and is not mentioned in the paper.

## One-seed strength calibration

The new calibration reruns the complete frozen strength grid with one common
regression learning rate and protocol-specific long budgets:

| Protocol | Grouping | Strengths | Updates |
|---|---|---|---:|
| Fixed IC | grouped channels | `0.003`, `0.006`, `0.01`, `0.03`, `0.06`, `0.09` | 35,000 |
| Fixed IC | separate channels | `0.0015`, `0.003`, `0.006`, `0.01`, `0.02`, `0.03` | 35,000 |
| Varying IC | grouped channels | `0.003`, `0.008`, `0.025`, `0.04`, `0.06` | 50,000 |
| Varying IC | separate channels | `0.003`, `0.008`, `0.025`, `0.04`, `0.06` | 50,000 |

All 22 runs use regression learning rate `2e-4` and apprentice seed `600601`.
This explicitly retains the Fixed strengths `0.01` and `0.03`, even though
they did not enter the Pareto set in the previous analysis. Every run starts
from the same paired apprentice initialization and uses the same batch-order
RNG. Run identities include phase, learning rate, budget, protocol, grouping,
strength, and seed, so old short-budget results cannot be mixed with this
homogeneous block.

All runs evaluate native GO sparsity only, with autoregressive validation every
25 updates starting at update 0. Hard-threshold candidates and teacher-forced
diagnostics are omitted. The training worker verifies test-corpus coverage and
expert provenance but does not evaluate test samples.

### Server launch

Preview the complete launch from the project root:

```bash
bash Revision/GO_Sensitivity/launch_tmux.sh --preview
```

Then start all 22 workers in parallel:

```bash
bash Revision/GO_Sensitivity/launch_tmux.sh
```

The launcher creates one detached tmux session per strength and combination,
for example `p6_cal_fixed_gc_s0p003`. Each session writes its own log below
`results/strength_calibration/logs/` and closes automatically when its Julia
worker exits. Active sessions with the same name are not duplicated. A
completed run is skipped, while an interrupted run resumes from
`resume/latest.jld2`. As in the other server studies, `systemd-inhibit` is used
by default; `--no-systemd-inhibit` is available for intentional local/debug
use. `--protocol` and `--grouping` can restrict a restart to a subset.

The result root can be overridden with `P6_CALIBRATION_RESULTS_DIR`. The local
plain-Julia fallback remains available and processes the same 22 jobs
sequentially:

```powershell
julia --project=. .\Revision\GO_Sensitivity\run_strength_calibration_pilot.jl
```

After the complete server result directory has been copied locally, run:

```powershell
julia --project=. .\Revision\GO_Sensitivity\inspect_strength_calibration_pilot.jl
```

The inspector accepts only complete 35,000/50,000-update blocks with learning
rate `2e-4` and the exact frozen strength lists. It creates one validation SVG
per combination below `results/strength_calibration/analysis/`; old
short-budget runs are ignored. Every validation checkpoint is an unconnected
strength-colored circle. Larger black-outlined diamonds mark the pooled
nondominated checkpoints. The y-axis is logarithmic.

Relocated server results may contain an absolute server-side `expert_path`.
When that path is unavailable, the closed-loop worker falls back to the local
`Expert_Apprentice_Distillation/experts/<protocol>/agent.jld2` checkpoint and
requires its SHA-256 identifier to match the expert recorded in every run.

By default, and only after all four long-budget blocks are complete, the
inspector evaluates every retained Pareto candidate in closed loop together
with the MAT expert. Fixed IC uses the shared 200-step episode. Varying IC uses
the eight predeclared test episodes. Evaluations are cached by expert,
candidate, and test case.

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

### Calibration test-split rule

The calibration diagnostic did not select candidates or change training. Its
result files remain marked `scientific_selection_allowed = false`. Production
strengths were fixed from the offline loss/validation behavior, and Package-6
production selection is performed only on validation matching and native SC
sparsity. The existing test episodes are therefore used only as terminal,
selection-inert checks; observing their returns must not trigger a later
strength, checkpoint, mask, or training decision.

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
