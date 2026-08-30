# Package 7 — Fixed-IC Apprentice comparison

Package 7 trains GO, GR, Group Lasso and GrOWL with grouped-channel (GC) and
separate-channel (SC) grouping. Every configuration uses three strengths and
three paired seeds (nine training workers).
It is independent of Package 6.

## Server commands

Preview the default matrix:

```bash
bash Revision/Package7/launch_tmux.sh --preview
```

Start all eight configured strength grids (72 training and eight analysis tmux
sessions):

```bash
bash Revision/Package7/launch_tmux.sh
```

Start one configuration at its configured three-strength grid (nine training
workers and one analyzer):

```bash
bash Revision/Package7/launch_tmux.sh --config go-sc
```

Repeated `--strength` replaces the configured grid for one launch:

```bash
bash Revision/Package7/launch_tmux.sh --config go-sc --strength 0.01

bash Revision/Package7/launch_tmux.sh \
  --config go-sc \
  --strength 0.006 \
  --strength 0.015 \
  --strength 0.0375
```

Each new launch automatically receives a short UTC timestamp ID such as
`260830_120000`. Use that ID to resume or rerun only the analyzer:

```bash
bash Revision/Package7/launch_tmux.sh \
  --config go-sc \
  --experiment-id 260830_120000 \
  --analysis-only
```

Use `--retry-failed` to resume failed runs and `--results-dir PATH` to override
the result root.
The launcher uses no `systemd-inhibit` wrapper.

The editable grids live only in `P7_STRENGTH_GRIDS` in `Package7Study.jl`.
Every grid uses a factor of 2.5 between adjacent strengths.

## Result layout

Each run is stored below:

```text
results/<timestamp>/<configuration>/<strength-tag>/r01..r03/
```

After each run, atomic evaluation shards are consolidated into
`evaluations.jld2`. One analyzer waits for all nine runs of a configuration and
writes their pooled analysis to:

```text
results/<timestamp>/<configuration>/analysis/
  pareto_all_points.svg
  pareto_all_points.pdf
  evaluations.csv
  pooled_pareto_front.csv
  selected_test_candidate.jld2
  test/
    test_results.jld2
    test_episode.csv
    test_curves.svg
```

The plot contains every 25-update evaluation from all three strengths and all
three seeds, with four threshold colors and the pooled Pareto front. The hover
data identify the strength; the x-axis shows active groups on a linear scale.
`evaluations.csv` contains the native point from every checkpoint plus only
those hard-threshold points that reduce the active-group count. Hard thresholds
without a group-count reduction are skipped before validation. In contrast,
`pooled_pareto_front.csv` contains only the pooled non-dominated points and
marks whether each point satisfies the Fixed-IC quality criterion
`validation_matching <= 0.01`. The same threshold is drawn as a dashed line in
the Pareto plot. No duplicate JLD2 analysis product is written.

After freezing the pooled validation results, the analyzer selects the point
with the fewest `active_inputs` among those satisfying
`validation_matching <= 0.01`. Ties use active groups, validation MSE, update,
run ID and candidate ID. The selected checkpoint is evaluated once on the
Fixed test episode with its stored input mask. `test_results.jld2` contains the
200 actions, environment rewards and direct per-step `state_Nu` values plus
their aggregate scores. Test data never influence the selection.

## Paper artifacts

After all eight analyzers have completed, build the paper inputs with the
experiment directory name:

```bash
julia --startup-file=no --project=. Revision/Package7/make_paper_figures.jl \
  --experiment-id 260830_155658
```

When `--experiment-id` is omitted, the newest direct experiment directory in
`Revision/Package7/results` is selected automatically:

```bash
julia --startup-file=no --project=. Revision/Package7/make_paper_figures.jl
```

From a Julia REPL, the equivalent call is:

```julia
include("Revision/Package7/make_paper_figures.jl")
main(["--experiment-id", "260830_155658"])
```

The script only reads frozen analysis, test, and Fixed baseline artifacts. It
writes the complete `figure_1_selected_sensor_masks.{svg,pdf}`, two additional
2×2 mask figures for GO/GR and Group-Lasso/GrOWL, the corresponding selected-
candidate table as CSV/Markdown, `figure_s1_pareto_comparison.{svg,pdf}`, a
compact `paper_metrics.jld2`, and a SHA-256 provenance manifest below
`results/<experiment-id>/paper/`. Use `--check-only` to validate all inputs
without exporting figures.
