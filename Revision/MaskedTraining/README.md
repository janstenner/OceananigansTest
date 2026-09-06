# Package 11: direct MAT training with fixed sensor masks

Forty new MAT trainings at **Ra = 1e4**: ten runs for each Fixed/Varying × GC/SC
combination. Agents start from the same random initialization as their dense
MAT references, not from apprentice or expert checkpoints. The unchanged
Comparison MAT results are reused as dense references.

## Frozen inputs

The launcher reads the existing `MAT_IPPO_Comparison/results/run_plan.jld2`.
It requires exactly ten distinct run/IC seed pairs and complete dense MAT
references for both protocols. It copies the complete Varying training IC
sequences rather than drawing new ones. Training budgets and MAT configuration
are inherited from the Comparison: 2,000 Fixed episodes and 4,000 Varying
episodes, `modified_full`, and unchanged network sizes/hyperparameters.

Among the four frozen Package-7/8 candidates per grouping, selection minimizes
`active_inputs`, then validation MSE. Test performance is never used.
The current source artifacts select:

| Protocol | Grouping | Candidate | Active groups | Validation MSE |
|---|---|---|---:|---:|
| Fixed | GC | go-gc | 1/32 | 0.0001651572 |
| Fixed | SC | go-sc | 1/96 | 0.0002668091 |
| Varying | GC | go-gc | 1/32 | 0.0003060061 |
| Varying | SC | gr-sc | 1/96 | 0.0017644344 |

`manifest.jld2` freezes the masks, candidate-selection audit, source checkpoint
hashes, ten seed pairs, full IC sequences, dense reference identities, run-file
and RL source hashes, and corpus identity. Workers reject changed sources and
check initial parameter hashes, including the distinct Package-3 import hash
format. Imported reference files lack a stored Ra parameter; their source is
the standard Comparison/Package-3 experiment. The new workers explicitly check
the loaded standard Revision run file has Ra=1e4.

## Launch on the server

From the project root:

```bash
bash Revision/MaskedTraining/launch_tmux.sh --preview
bash Revision/MaskedTraining/launch_tmux.sh
```

The first full launch starts **40 detached tmux sessions**, one per training
run. There is no analysis worker or dense retraining. Sessions survive SSH
disconnects, close on completion, and write independent logs and a TSV job
manifest under `results/launches/<launch-id>/`. OpenBLAS/OpenMP defaults are
3/1 threads, following the other Revision launchers. `JULIA_BIN` overrides Julia.

Optional filters: `--protocol fixed|varying`, `--grouping gc|sc`.
Source overrides: `--comparison-dir`, `--package7-results`, `--package8-results`.
Output override: `--results-dir` (or `PACKAGE11_RESULTS_DIR`).
`--preview` validates the actual sources and prints the commands without
persisting the experiment manifest or starting sessions.

Restart using the same command. Completed matching results and active sessions
are skipped. Failed/interrupted runs restart from the original seed; there is
no mid-training resume checkpoint. Per-result lock directories prevent duplicate
workers. After a hard kill, remove a stale lock only after verifying its recorded
PID/host has no running worker. A changed frozen configuration requires a new
result directory. Prepare the manifest on the machine where training will run;
it records absolute source paths.

## Observation and reward separation

`MaskedTraining.jl` follows the Comparison training-loop stage order.
`with_masked_state` temporarily exposes `full_state .* mask` to every agent
callback. Both PPO `state` and `next_state`, including terminal bootstrap
observations, are masked. It restores the original full state in `finally`.
Environment steps, featurization and hook callbacks receive full observations;
the reward uses the complete physical sensor tensor with the unchanged reward
function. Masks remain fixed throughout training. The global Nusselt reward
is not recomputed from sparse observations.

Atomic results at `results/runs/<protocol>/<grouping>/<run-id>.jld2` retain the
Comparison result schema (agent, rewards, per-step rewards, failures, runtime,
seeds, initial/final hashes and observed IC trace), plus the mask provenance,
dense reference, control-step count and throughput. Exceptions have separate
`.failure.jld2` diagnostics and do not mark a run complete.

This implementation prepares and executes the requested training experiment.
Aggregate plots and final deterministic validation/test evaluation remain a
separate follow-up; use the same Comparison validation cases and the existing
one Fixed/eight Varying Revision test cases, with masked deterministic actions
and full-sensor rewards. Do not reselect masks from these results.

## Checks

```bash
julia --startup-file=no --project=. Revision/MaskedTraining/test_selection.jl
julia --startup-file=no --project=. Revision/MaskedTraining/test_masked_training.jl
julia --startup-file=no --project=. Revision/MaskedTraining/test_runtime.jl fixed
julia --startup-file=no --project=. Revision/MaskedTraining/test_runtime.jl varying
julia --startup-file=no --project=. Revision/MaskedTraining/test_worker.jl fixed
julia --startup-file=no --project=. Revision/MaskedTraining/test_worker.jl varying
bash -n Revision/MaskedTraining/launch_tmux.sh
```

The runtime smoke uses two physical control steps and one PPO update in a fresh
process, with the production seed/mask/initialization. It writes no production
result. The zero-episode worker tests also verify generated Comparison seeds,
atomic saving, frozen identities and completed-run reuse.
`run_worker.jl --episodes N` is only a pipeline smoke override; use a
separate prepared result directory, since normal launchers reject short results.
