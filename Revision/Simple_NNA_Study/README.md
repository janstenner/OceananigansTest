# Simple NNA Study: Varying-IC sparsity comparison

This study is the dense-layer counterpart to Package 8. It uses the same
Varying-IC train, validation, and terminal test splits, seeds, update budget,
batches, validation cadence, mask thresholds, GO/GR strengths, and GC/SC group
definitions. The apprentice architecture, reduced four-configuration matrix,
and stricter validation-quality threshold differ.

## Apprentice architecture

The apprentice mean network has exactly the Varying-IC IPPO actor layout:

```text
Dense(360, 102, gelu)
Dense(102, 102, gelu)
Dense(102, 1, identity)
```

It is shared over all twelve actuator columns, as in parameter-sharing IPPO.
The scalar `logσ` parameter is retained for parameter accounting and checkpoint
compatibility but is not used by the deterministic distillation loss.

The MAT apprentice contains 75,551 trainable parameters in total. Its actor
(observation encoder plus action decoder, excluding the separate critic
encoder and value head) contains 47,698. The unchanged IPPO width rule
`hidden = floor(10 * nna_scale)` cannot equal 47,698 exactly. The closest width
is 102, hence `nna_scale = 10.2`, with 47,432 trainable actor parameters
including `logσ` (266 fewer, or 0.56%). These values are asserted at worker
startup and stored in every manifest and run configuration.

## Package-8-derived settings

- master seed `20_260_851` and the same three apprentice/batch seed pairs;
- 100,000 updates, batch size 100, learning rate `2e-4`;
- validation batch size 512 every 25 updates, including update zero;
- absolute mask thresholds `(0.0, 0.003, 0.006, 0.012)` with max-input-L1
  group importance;
- Pareto objectives `active_inputs` and Varying validation MSE;
- stricter quality threshold `validation MSE <= 0.02` (Package 8 used `0.03`);
- a GR-SC-only diagnostic terminal-test sweep over every pooled-Pareto
  candidate from the sparsest `validation MSE <= 0.02` selection through the
  candidate with 17 active groups, inclusive;
- no test data enter strength, checkpoint, threshold, or mask selection.

The four configurations and Package-8 strength grids are:

| Configuration | Strengths |
|---|---|
| `go-gc` | `0.008 / 0.02 / 0.05` |
| `go-sc` | `0.016 / 0.04 / 0.1` |
| `gr-gc` | `0.000024 / 0.00006 / 0.00015` |
| `gr-sc` | `0.000048 / 0.00012 / 0.0003` |

## Four tmux launches

Use one shared experiment ID and invoke the launcher once per configuration.
Each call starts nine training workers and one analyzer, for 40 sessions over
the four calls:

```bash
EXPERIMENT_ID="simple_nna_$(date -u +%y%m%d_%H%M%S)"

bash Revision/Simple_NNA_Study/launch_tmux.sh --experiment-id "$EXPERIMENT_ID" --config go-gc
bash Revision/Simple_NNA_Study/launch_tmux.sh --experiment-id "$EXPERIMENT_ID" --config go-sc
bash Revision/Simple_NNA_Study/launch_tmux.sh --experiment-id "$EXPERIMENT_ID" --config gr-gc
bash Revision/Simple_NNA_Study/launch_tmux.sh --experiment-id "$EXPERIMENT_ID" --config gr-sc
```

`--config` is required. `--preview`, `--retry-failed`, `--results-dir`,
`--openblas-threads`, and `--omp-threads` behave as in Package 8. Repeated
`--strength` or `--threshold` arguments replace the selected configuration's
defaults for that call; native threshold `0.0` remains automatic.

To rerun one analyzer for an existing experiment:

```bash
bash Revision/Simple_NNA_Study/launch_tmux.sh \
  --analysis-only --experiment-id "$EXPERIMENT_ID" --config go-gc
```

All analyzers use the single quality threshold `0.02`. In addition, the
`gr-sc` analyzer tests every pooled-Pareto candidate in ascending group count
from its sparsest quality-qualified point through 17 active groups. Analysis
outputs use `selected_test_candidates.jld2` and numbered
`test/candidate_XX/` directories.
The paper mask figure uses the 17-group sweep candidate in the GR-SC panel;
the other panels continue to show their quality-selected candidates.

## Outputs

Training retains atomic resume state and Pareto checkpoints. Each analyzer
writes consolidated evaluation CSV/JLD2 files, Pareto SVG/PDF files, and the
validation-frozen test candidate set. The terminal test covers the same eight
deterministic Varying test cases as Package 8 and preserves split, basis seed,
mirror, offset, evaluation seed, episode, control step, simulation time,
actions, rewards, and direct `state_Nu` values.

After all four analyzers are complete:

```bash
julia --startup-file=no --project=. Revision/Simple_NNA_Study/make_paper_figures.jl \
  --experiment-id "$EXPERIMENT_ID"
```
