# Higher-Rayleigh-Number Study

## Scope

This directory will contain the complete expert-to-sparse-controller workflow
for the two higher-Rayleigh-number Varying-IC systems:

- `Ra = 5e4` (`ra5e4`);
- `Ra = 1e5` (`ra1e5`).

The final directory should be self-contained with respect to all
higher-Rayleigh-number study scripts and generated study data. Existing MAT
training results remain the immutable upstream source, while selected compact
experts, their test baselines, both distillation corpora, GO/GR training
artifacts, analyses, and final evaluation results live below
`Revision/Higher_Ra_Study`.

## Implemented first stage: MAT expert extraction

`analyze_runs.jl` performs the complete first stage for both Rayleigh
numbers:

1. Recursively scan the corresponding `MAT_expert_training` result root for
   available global best-so-far, completed final, and current resume
   checkpoints.
2. Require at least 100 completed episode rewards and recompute the mean of the
   latest 100 rewards for every candidate checkpoint.
3. Select the checkpoint with the largest rolling-100 mean independently for
   `ra5e4` and `ra1e5`.
4. Validate and byte-copy the matching compact `varying/expert.jld2` already
   published beside a selected best-so-far checkpoint. If no matching compact
   source exists, publish one from the selected checkpoint. The resulting
   `experts/<ra>/expert.jld2` contains only the agent; its trajectory is empty
   and every trajectory backing buffer has capacity one, matching the existing
   Revision distillation experts.
5. Save MAT-IPPO-style SVG learning curves for each Rayleigh number: a
   median/IQR aggregate with translucent individual runs and a separate
   individual-run figure. The displayed rolling window is 100 episodes, equal
   to the expert-selection metric.
6. Evaluate each selected expert deterministically on all eight cases of its
   own higher-Ra test split for 200 control steps. Results, reward histories,
   full-state `state_Nu`, actions, source hashes, and aggregate metrics are
   stored in `Baselines/<ra>/expert.jld2`.
7. Record selection, expert, plot, source-checkpoint, and baseline provenance
   in `experts/selection_manifest.jld2`.

Default invocation:

```bash
julia --startup-file=no --project=. Revision/Higher_Ra_Study/analyze_runs.jl
```

Alternative result roots can be supplied with `--ra5e4-results` and
`--ra1e5-results`. The script can be rerun while training is still in progress;
atomic resume and best-so-far checkpoints are treated as available snapshots.
A later rerun may legitimately replace the selected expert if a better
rolling-100 candidate has appeared.

While only one result root is locally available, `--study ra5e4` or
`--study ra1e5` restricts the complete extraction, plotting, and baseline
workflow to that Rayleigh number. The default `--study all` analyzes both.

## Planned local layout

```text
Higher_Ra_Study/
  analyze_runs.jl
  Higher_Ra_Study.md
  experts/
    selection_manifest.jld2
    ra5e4/expert.jld2
    ra1e5/expert.jld2
  Baselines/
    ra5e4/expert.jld2
    ra1e5/expert.jld2
  plots/
  Distillation_Corpuses/
    ra5e4/
    ra1e5/
  GO_GR_Study/
    manifests/
    results/
    analysis/
```

## Planned second stage: separate distillation corpora

Each Rayleigh number requires its own expert-rollout distillation corpus. The
corpora must never mix states, actions, or expert identities across Rayleigh
numbers. The planned implementation will therefore add shared higher-Ra corpus
utilities plus one manifest and worker grid per `ra5e4` and `ra1e5`.

The design should follow the established Varying-IC distillation protocol:

- use only the selected expert for the matching Rayleigh number;
- freeze the expert SHA-256, run-file SHA-256, state-corpus SHA-256, split, and
  case plan before generation;
- retain the compact global `3 x 48 x 8` observations and deterministic expert
  action means;
- preserve train/validation/test isolation;
- cover the established basis-snapshot, mirror, and horizontal-offset plan;
- store one atomic file per worker-owned shard and support restart without
  rewriting valid shards;
- keep all generated higher-Ra corpus files below
  `Higher_Ra_Study/Distillation_Corpuses/<ra>/`.

The exact number of training shards and whether the existing offset set should
be reduced for computational cost must be frozen before corpus generation.

## Planned third stage: GO and GR apprentice studies

After both corpora are complete, this directory will receive a paired GO/GR
study modeled on Package 8. The implementation should contain:

- a shared study module defining both Rayleigh numbers, GO/GR configurations,
  seeds, regularization strengths, checkpoint schedules, and atomic paths;
- manifest preparation that verifies the local expert and matching local
  distillation-corpus identities;
- one restart-safe training worker per Rayleigh number, method, strength, and
  replicate;
- a tmux launcher with Rayleigh-number, method, strength, and replicate
  filters;
- validation-only Pareto construction over active sensor groups/inputs and
  autoregressive validation MSE;
- frozen candidate selection before any terminal test-set rollout;
- deterministic closed-loop test evaluation on the matching eight higher-Ra
  test cases;
- Package-8-style aggregate metrics, learning/Pareto plots, paper artifacts,
  and provenance manifests.

GO and GR should be compared under paired initialization, corpus batches, and
evaluation cases wherever the algorithms permit it. Noise robustness is not
part of this initial higher-Ra plan and should only be added as a separately
specified post-hoc study.

## Decisions still to freeze before stages two and three

- Whether the distillation corpora reproduce the full Package-8 training
  offset coverage or use a smaller cost-controlled grid.
- The GO and GR regularization-strength grids at each Rayleigh number.
- Training budgets and checkpoint cadence, which may need to grow with the
  higher-Rayleigh dynamics.
- The number of paired training seeds.
- The candidate-selection rule when several points share the minimum active
  input count.
- Whether an intermediate low-validation-MSE candidate, analogous to
  `C_match`, should be frozen in addition to the sparsest selected apprentice.
