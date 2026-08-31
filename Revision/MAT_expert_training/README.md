# MAT expert training

This directory continues all ten validation-ranked final MAT checkpoints for
Fixed IC and Varying IC. It is a separate expert-production
experiment and does not alter the MAT stability or MAT-IPPO comparison runs.

## Ra = 10^5 Varying-IC training

The independent Ra=10^5 path starts fresh MAT agents with the dedicated
`VaryingIC_MAT_Ra1e5.jl` run file and `varying_ic_corpus_Ra1e5.jld2`. It does
not depend on the Ra=10^4 MAT-IPPO ranking. For a fixed training budget:

```bash
bash Revision/MAT_expert_training/launch_tmux_ra1e5.sh \
  --runs 10 --episodes 4000
```

For a shared threshold stop:

```bash
bash Revision/MAT_expert_training/launch_tmux_ra1e5.sh \
  --runs 10 --threshold -610.0
```

In threshold mode, the first worker whose mean over the latest 100 completed
episode rewards is strictly greater than the supplied value requests the stop.
Every other worker finishes its active episode and saves before exiting. In
episode mode, every worker independently reaches the requested episode count.

The launcher defaults to master seed `20260901`, three OpenBLAS threads, and one
OpenMP thread per worker. `--master-seed`, `--openblas-threads`,
`--omp-threads`, and `--results-dir` override these defaults. Use `--preview`
to inspect the deterministic Run/IC seed plan before the corpus is complete.

Results default to `results_ra1e5`. `experiment_manifest.jld2` freezes the
corpus, run-file hashes, seed plan, and stop configuration. Every worker writes
an atomic `resume/latest.jld2` after every episode and a `final.jld2` on normal
completion. Across all workers, `varying/best_so_far.jld2` and the compact,
metadata-free `varying/expert.jld2` always represent the largest rolling-100
reward mean observed so far. Rerunning the identical launch command resumes
incomplete workers.

## Stop rules

- Fixed IC: the loaded source/resume policy and then the current policy after
  every newly completed training episode are evaluated once with deterministic
  mean actions. `state_Nu(env)` is measured from the complete state at all 200
  evaluation steps. The first worker whose
  negative full-state Nusselt sum, `-sum(state_Nu)`, is strictly greater than
  `-555.0` becomes the Fixed expert candidate. This is equivalent to a mean
  full-state global Nusselt number below `2.775` and is directly comparable to
  the deterministic old-expert rollout from which the threshold was derived.
- Varying IC: after every newly completed episode, the mean of the latest 100
  episode rewards is calculated exactly as in `randomIC/randomIC_MAT.jl`. The
  first worker with a mean strictly greater than `-610.0` becomes the Varying
  expert candidate.

The winner writes an atomic protocol stop signal. The other nine workers never
abort an active episode: they finish it, atomically save their resume and final
checkpoints, and then exit. The original Package-4 files are read-only.

Independently of the thresholds, every worker also competes for one atomic
protocol-wide checkpoint:

```text
results/fixed/best_so_far.jld2
results/fixed/expert.jld2
results/varying/best_so_far.jld2
results/varying/expert.jld2
```

For Fixed IC it is replaced only by a higher negative full-state Nusselt sum
from the latest deterministic evaluation.
For Varying IC it is replaced only by a higher mean over the latest 100
completed episode rewards. The checkpoint contains the complete resumable
agent, histories, seeds, continuation trace, metric, and provenance. Alongside
it, `expert.jld2` is updated from a separately loaded copy of the same agent. It
contains only the `agent` key and has an empty trajectory whose backing buffers
have capacity one, exactly like the final threshold expert exports. The live
training agent is never compacted. Existing Package-4 checkpoints do not store
full-state Nusselt curves, so each loaded Fixed source/resume policy is
evaluated once at worker startup. Both protocols therefore produce corrected
best-so-far files before the first continuation episode.

## Preview and launch

From the repository root, validate all twenty selected checkpoint identities and
preview exactly 21 tmux sessions:

```bash
bash Revision/MAT_expert_training/launch_tmux.sh --preview
```

The validation identity check permits an absolute score difference of `1e-5`
for last-digit Julia/CUDA/hardware variation. Both the frozen and observed
scores and their delta are recorded; run ID, protocol, seeds, episode count,
algorithm, status, and checkpoint hashes are still checked exactly.

Start the 20 training workers and one waiting test/export worker:

```bash
bash Revision/MAT_expert_training/launch_tmux.sh
```

This revision freezes the newly collected Package-4 ranking. A server result
root that already contains the previous `selection_manifest.jld2` is
intentionally rejected. Preserve old results and start the rerun in a fresh
root, for example:

```bash
bash Revision/MAT_expert_training/launch_tmux.sh \
  --results-dir Revision/MAT_expert_training/results/rerun_20260821
```

Use the same `--results-dir` for a later manual cutoff.

The sessions are:

```text
pmat_fixed_01 ... pmat_fixed_10
pmat_varying_01 ... pmat_varying_10
pmat_test
```

Each session closes after its process finishes. `JULIA_BIN`,
`MAT_EXPERT_RESULTS_DIR`, and
`MAT_IPPO_RESULTS_DIR` override the executable and result roots.
`DISTILLATION_EXPERTS_DIR` overrides the tracked expert destination.

## Monitoring

Each launch stores `jobs.tsv` and one persistent log per session below:

```text
results/launches/<launch-id>/
```

Useful commands are:

```bash
tmux ls
tmux attach -t pmat_fixed_01
tail -f Revision/MAT_expert_training/results/launches/<launch-id>/pmat_fixed_01.log
tail -f Revision/MAT_expert_training/results/launches/<launch-id>/pmat_test.log
```

Every completed training episode prints protocol, rank, run ID, additional and
total episode count, sensor-derived episode reward, mean full-state global
Nusselt number where applicable, current stop metric, target, and whether the
criterion was reached. `global_best_updated=true` marks a new
protocol-wide top candidate. A resume checkpoint is atomically replaced after
every completed episode.

## Manual best-so-far cutoff

If a threshold is not reached in the desired wall-clock budget, do not kill the
training sessions first. From the repository root, freeze the current global
best for both protocols and request a graceful episode-boundary stop with:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_expert_training/finalize_best_so_far.jl --protocol all
```

`--protocol fixed` and `--protocol varying` can be used independently. The
command copies each current `best_so_far.jld2` to an immutable
`manual_candidate.jld2`, publishes the normal stop signal, and leaves active
workers enough time to finish their current episode and save final checkpoints.
The already running `pmat_test` session then evaluates and exports the frozen
best snapshot through the same test/publication pipeline as a threshold winner.
If a threshold winner already exists, the command preserves it.

## Test evaluation

`pmat_test` waits until a protocol selection and all ten final checkpoints for
that protocol exist. The selection is either the first threshold winner or an
explicitly frozen manual best-so-far snapshot. It then evaluates the selected
agent with deterministic mean actions:

- Fixed IC: the one shared 200-step episode;
- Varying IC: all eight predefined test episodes from two test bases, both
  mirror choices, and offsets 0 and 20.

Every test step stores both the sensor-derived environment reward and the
full-state global Nusselt number from `state_Nu(env)`. Episode caches,
`scores.csv`, and `summary.jld2` include the reward return, `-sum(state_Nu)`,
and mean global Nusselt number. Separate SVG/PNG reward and full-state Nusselt
curves are written below `results/test/fixed` and `results/test/varying`. Test
results do not alter the winner selection.

To evaluate the unchanged validation-rank-1 Package-4 MAT checkpoints directly,
without starting continuation training, run each protocol in a fresh Julia
process:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_expert_training/evaluate_package4_candidate.jl --protocol fixed
julia --startup-file=no --project=. \
  Revision/MAT_expert_training/evaluate_package4_candidate.jl --protocol varying
```

The output below `results/package4_source_candidate_test/<protocol>` contains
the raw `state_Nu(env)` value and environment reward for every test step, one
full-state Nusselt curve per episode, and CSV/JLD2 summaries. The Varying
evaluation additionally writes a box plot of `sum(state_Nu(env))` across its
eight predefined test episodes. Cached episodes are accepted only when their
checkpoint SHA-256 matches the currently frozen rank-1 source checkpoint.

After downloading an expert-training threshold result, evaluate the selected
checkpoint with the identical test protocol using:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_expert_training/evaluate_package4_candidate.jl \
  --protocol varying --selection threshold
```

This reads `results/varying/candidate.jld2`, verifies that it is a real
threshold selection, cross-checks it against `results/varying/best_so_far.jld2`,
and writes the separate comparison output below
`results/threshold_candidate_test/varying`.

Once the Fixed source evaluation and Varying threshold result are locally
available, publish this final hybrid selection to Expert-Apprentice
Distillation with:

```bash
julia --startup-file=no --project=. \
  Revision/MAT_expert_training/publish_selected_experts.jl
```

The publisher creates `results/fixed/expert.jld2` from the unchanged
Package-4 Fixed rank-1 checkpoint, verifies the existing compact Varying expert
against the threshold checkpoint, stages and validates both agent-only files,
and then replaces both tracked Distillation experts.

After both protocol tests complete, the worker empties and shrinks each winner
agent's trajectory exactly as for the existing Distillation experts. The
agent-only exports are written to
`results/experts/fixed/expert.jld2` and
`results/experts/varying/expert.jld2`. Only after both exports have been staged
and reloaded successfully are the tracked files
`Expert_Apprentice_Distillation/experts/fixed/agent.jld2` and
`Expert_Apprentice_Distillation/experts/varying/agent.jld2` replaced. A
`results/experts/publication_manifest.jld2` records winner and SHA-256
provenance.
