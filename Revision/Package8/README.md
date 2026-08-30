# Package 8: Varying-IC regularizer comparison

Package 8 is the Varying-IC counterpart to Package 7. It trains the eight
regularizer/grouping combinations on the Varying training corpus, evaluates
expert-action matching only on the independent Varying validation corpus, and
runs the frozen sparsest qualified candidate on the Varying test split.

## Protocol

- independent master seed `20_260_851` and three paired apprentice/batch seeds;
- 100,000 updates, training batch size 100, learning rate `2e-4`;
- validation batch size 512 every 25 updates, including update zero;
- absolute mask thresholds `(0.0, 0.003, 0.006, 0.012)` with max-input-L1
  group importance;
- Pareto objectives: `active_inputs` and Varying validation MSE;
- quality threshold: validation MSE `<= 0.03`;
- no test data enter strength, checkpoint, threshold, or mask selection.

The numerical strength grids initially equal the current Package-7 grids.
Their comments in `Package8Study.jl` record the Varying inherited defaults from
`GrOWL/MAT_expert_apprentice.jl`.

## Launching

All eight configurations (72 training workers and eight analyzers):

```bash
bash Revision/Package8/launch_tmux.sh
```

One configuration with its configured three-strength grid:

```bash
bash Revision/Package8/launch_tmux.sh --config go-sc
```

Explicit strengths replace that grid:

```bash
bash Revision/Package8/launch_tmux.sh --config go-sc \
  --strength 0.01 --strength 0.025 --strength 0.0625
```

Only rerun analyzers for an existing experiment:

```bash
bash Revision/Package8/launch_tmux.sh \
  --analysis-only --experiment-id YYMMDD_HHMMSS
```

`--preview`, `--retry-failed`, `--results-dir`, `--openblas-threads`, and
`--omp-threads` behave as in Package 7. The BLAS/OpenMP defaults are 3/1.

## Outputs

Training retains atomic resume state and Pareto checkpoints. Evaluation shards
are consolidated into one verified `evaluations.jld2` per completed run. Each
configuration analyzer writes `evaluations.csv`, `pooled_pareto_front.csv`, the
Pareto SVG/PDF, and a validation-frozen test candidate.

The terminal test covers all eight deterministic Varying test cases (two base
states, both mirror choices, and offsets 0/20). `test_results.jld2` and
`test_episodes.csv` retain split, base seed, mirror, offset, evaluation seed,
episode, control step, simulation time, actions, rewards, and direct
`state_Nu` values.

After all eight configuration analyzers are complete, paper artifacts can be
generated for an explicit experiment or the newest result directory:

```bash
julia --startup-file=no --project=. Revision/Package8/make_paper_figures.jl \
  --experiment-id YYMMDD_HHMMSS

julia --startup-file=no --project=. Revision/Package8/make_paper_figures.jl
```
