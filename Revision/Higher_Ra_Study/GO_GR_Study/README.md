# Higher-Ra GO/GR study

This directory implements the Package-8-style sparse-apprentice study for the
complete `Ra=5e4` and `Ra=1e5` expert-rollout corpora. Only Group Ordered (GO)
and Group Reweighted (GR) training are included, each with grouped-channel
(GC) and separate-channel (SC) sensor grouping. Group Lasso and GrOWL are not
part of this study.

## Frozen protocol

- 100,000 regularized updates, no post-pruning fine-tuning;
- batch size 100 and regression learning rate `2e-4`;
- autoregressive validation with batch size 512 every 25 updates, including
  update zero;
- three paired apprentice/batch-seed replicates per strength;
- absolute mask thresholds `(0.0, 0.003, 0.006, 0.012)` with max-input-L1
  importance and at least one retained group;
- Pareto objectives: global active inputs and validation expert-action MSE;
- quality thresholds `(0.03, 0.015, 0.0075)`;
- no test data are used for training, Pareto construction, or candidate
  selection.

The strength grids are:

```text
go-gc: 0.00128,    0.0032,    0.008,    0.02,    0.05
go-sc: 0.00256,    0.0064,    0.016,    0.04,    0.1
gr-gc: 0.00000384, 0.0000096, 0.000024, 0.00006, 0.00015
gr-sc: 0.00000768, 0.0000192, 0.000048, 0.00012, 0.0003
```

Each Rayleigh number therefore has 60 training workers and four analysis/wait
workers. Within a configuration the analyzer pools all five strengths and all
three replicates. For each quality threshold it selects the point with the
fewest active inputs from the pooled validation Pareto front, followed by
active groups, validation MSE, update, run ID, and candidate ID as deterministic
tie-breakers. Missing quality levels produce no candidate; if several quality
levels select the same candidate, it is tested only once. Thus each
configuration evaluates zero to three frozen candidates.

Every selected candidate is tested on the matching eight Higher-Ra test cases
(two bases, both mirrors, offsets 0 and 20) for 200 control steps. Candidate
selection is saved atomically before the first test rollout.

## Launching

Start the complete `Ra=1e5` study:

```bash
bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra1e5.sh
```

Start the complete `Ra=5e4` study:

```bash
bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra5e4.sh
```

The launchers are independent files; there is no shared `common` launcher.
Each creates a timestamp experiment ID unless `--experiment-id` is supplied.
Use the same explicit ID for both launchers when the two Rayleigh numbers
should form one named experiment, while their result trees remain isolated by
`ra5e4/` and `ra1e5/`.

Useful filtered invocations:

```bash
bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra1e5.sh \
  --config go-sc

bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra5e4.sh \
  --preview

bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra1e5.sh \
  --analysis-only --experiment-id YYMMDD_HHMMSS
```

`--retry-failed`, `--results-dir`, `--openblas-threads`, and `--omp-threads`
follow Package 8. Repeated `--strength` values replace the grid for one
selected configuration. Repeated `--threshold` values replace the three
positive mask thresholds for one selected configuration; native `0.0` remains
automatic.

## Outputs

Results are stored below:

```text
GO_GR_Study/results/<ra>/<experiment-id>/
```

Every training run retains restart state, compact evaluations, and Pareto
candidate checkpoints. Each configuration analysis writes:

```text
<config>/analysis/
  status.jld2
  evaluations.csv
  pooled_pareto_front.csv
  pareto_all_points.svg
  pareto_all_points.pdf
  selected_test_candidates.jld2
  test/
    candidate_01/
      test_results.jld2
      test_episodes.csv
      test_curves.svg
    candidate_02/
    candidate_03/
```

Only directories for unique selected candidates are created.

## Paper artifacts

`make_paper_figures.jl` reads the completed analysis artifacts and the local
Higher-Ra baselines. It writes one independent paper directory per Rayleigh
number. Each directory contains a four-panel GO/GR × GC/SC Pareto figure, a
four-panel selected-mask figure, and a Markdown/CSV candidate table. The
Pareto panels show all three validation-MSE quality thresholds and every
distinct frozen candidate selected by them. A candidate selected by several
quality thresholds is shown and tabulated once with all corresponding
thresholds in its label.

By default, the script resolves the newest experiment independently below
each Rayleigh-number result root:

```bash
julia --startup-file=no --project=. \
  Revision/Higher_Ra_Study/GO_GR_Study/make_paper_figures.jl
```

For the named production experiment and one Rayleigh number:

```bash
julia --startup-file=no --project=. \
  Revision/Higher_Ra_Study/GO_GR_Study/make_paper_figures.jl \
  --study ra5e4 --experiment-id higher_ra_prod_01
```

The mask figure chooses, within every configuration, the fewest-input tested
candidate whose mean test-set `state_Nu` is no more than 5% above the expert
mean. Lower `state_Nu` is better. This fixed, predeclared tolerance can be
changed explicitly with `--near-expert-relative-tolerance`; the script errors
instead of silently choosing a candidate outside the requested band.

If an analysis worker failed, rerun only the four analyzers without restarting
training:

```bash
bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra5e4.sh \
  --analysis-only --retry-failed --experiment-id higher_ra_prod_01

bash Revision/Higher_Ra_Study/GO_GR_Study/launch_tmux_ra1e5.sh \
  --analysis-only --retry-failed --experiment-id higher_ra_prod_01
```
