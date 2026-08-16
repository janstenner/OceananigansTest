# MAT expert training

This directory continues all ten validation-ranked final MAT checkpoints for
Fixed IC and Varying IC. It is a separate expert-production
experiment and does not alter the MAT stability or MAT-IPPO comparison runs.

## Stop rules

- Fixed IC: the first worker whose newly completed episode reward is strictly
  greater than `-555.0` becomes the Fixed expert candidate.
- Varying IC: after every newly completed episode, the mean of the latest 100
  episode rewards is calculated exactly as in `randomIC/randomIC_MAT.jl`. The
  first worker with a mean strictly greater than `-610.0` becomes the Varying
  expert candidate.

The winner writes an atomic protocol stop signal. The other nine workers never
abort an active episode: they finish it, atomically save their resume and final
checkpoints, and then exit. The original Package-4 files are read-only.

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
total episode count, episode reward, current stop metric, target, and whether
the criterion was reached. A resume checkpoint is atomically replaced after
every completed episode.

## Test evaluation

`pmat_test` waits until a protocol winner and all ten final checkpoints for
that protocol exist. It then evaluates the winner with deterministic mean
actions:

- Fixed IC: the one shared 200-step episode;
- Varying IC: all eight predefined test episodes from two test bases, both
  mirror choices, and offsets 0 and 20.

Episode caches, `scores.csv`, `summary.jld2`, and SVG/PNG reward curves are
written below `results/test/fixed` and `results/test/varying`. Test results do
not alter the winner selection.

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
