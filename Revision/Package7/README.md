# Package 7 — Fixed-IC Apprentice comparison

Package 7 trains GO, GR, Group Lasso and GrOWL with grouped-channel (GC) and
separate-channel (SC) grouping. Every configuration uses three paired seeds.
It is independent of Package 6.

## Server commands

Preview the default matrix:

```bash
bash Revision/Package7/launch_tmux.sh --preview
```

Start all eight default configurations (24 training and eight analysis tmux
sessions):

```bash
bash Revision/Package7/launch_tmux.sh
```

Start one configuration at its default strength:

```bash
bash Revision/Package7/launch_tmux.sh --config go-sc
```

Start one explicit strength or several strength versions:

```bash
bash Revision/Package7/launch_tmux.sh --config go-sc --strength 0.09

bash Revision/Package7/launch_tmux.sh \
  --config go-sc \
  --strength 0.07 \
  --strength 0.09 \
  --strength 0.12
```

Use `--retry-failed` to resume failed runs, `--analysis-only` to regenerate or
wait only for analyses, and `--results-dir PATH` to override the result root.
The launcher uses no `systemd-inhibit` wrapper.

## Result layout

Each run is stored below:

```text
results/<configuration>/<strength-tag>/r01..r03/
```

After each run, atomic evaluation shards are consolidated into
`evaluations.jld2`. The fourth worker writes the pooled analysis to:

```text
results/<configuration>/<strength-tag>/analysis/
  pareto_all_points.svg
  pareto_all_points.pdf
  pareto_points.csv
  pareto_points.jld2
```

The plot contains every 25-update evaluation from all three seeds, with four
threshold colors and the pooled Pareto front.
