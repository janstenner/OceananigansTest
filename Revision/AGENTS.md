# AGENTS.md

## Purpose

This file documents the revision-specific implementation under `Revision/`.
The revision code supports the new experiments for the resubmission of the
Sparse Sensing paper.

## Directory Structure

- `Implementation_Plan.md`: Ordered implementation and experiment checklist for
  the paper revision.
- `package4.md`: Detailed agreed design for the dynamic, paired MAT-versus-IPPO
  experiment, Paket-3 import, validation, and result collection.
- `package6.md`: Detailed agreed workflow for the minimal GO sensitivity study
  on RBC teacher-rollout data.
- `package7_8.md`: Shared detailed workflow for the Fixed-IC and Varying-IC
  apprentice Pareto selection, threshold expansion, closed-loop validation,
  and final evaluation packages.
- `package10.md`: Frozen controller, channel-scale, white-noise, pairing,
  worker-ownership, persistence, and launch protocol for the Package-10 sensor-
  noise study.
- `Run_Files/FixedIC_MAT.jl` and `Run_Files/FixedIC_IPPO.jl`: Standalone
  fixed-initial-condition entry points.
- `Run_Files/VaryingIC_MAT.jl` and `Run_Files/VaryingIC_IPPO.jl`: Standalone
  varying-initial-condition entry points backed by the corpus.
- `Run_Files/README.md`: Public run-file configuration and orchestration notes.
- `Baselines/`: Four compact, parallel terminal-test reference runs for the
  Fixed/Varying MAT experts and zero-action controllers. Each protocol uses
  the same 200-step test cases as the revision studies and stores one atomic
  JLD2 file per controller with rewards, full-state `state_Nu`, actions,
  aggregate scores, and expert/test-data provenance.
- `MAT_Stability/`: Paired package-3 MAT workers, detached tmux launcher,
  atomic JLD2 results, and strict result collector.
- `MAT_Stability/README.md`: Server launch, restart, result, and collection
  instructions for package 3.
- `MAT_IPPO_Comparison/`: Dynamic seed/IC planning, Package-3 import, paired
  MAT/IPPO workers, persistent tmux launch, deterministic validation, and
  incremental collection for package 4.
- `MAT_IPPO_Comparison/README.md`: Server workflow, restart, import,
  validation, storage, and collection instructions for package 4.
- `MAT_expert_training/`: Validation-ranked all-ten MAT continuation for
  Fixed and Varying IC, protocol-wide episode-boundary stop coordination,
  atomic resume/final and protocol-wide full plus compact agent-only
  best-so-far checkpoints, explicit best-so-far manual cutoff, persistent
  21-session tmux launch,
  selected-candidate-only test-set sensor-reward and full-state global-Nusselt
  evaluation, direct Package-4/threshold-candidate evaluation, and hybrid
  compact publication of the unchanged Fixed rank-1 agent plus the Varying
  threshold winner into the tracked Expert-Apprentice Distillation
  checkpoints.
- `MAT_expert_training/README.md` and `PLAN.md`: Launch, monitoring, Fixed
  deterministic `-sum(state_Nu) > -555.0` and Varying rolling-100 reward
  `> -610.0` rules, exact
  selected source candidates, resume semantics, and test-result layout.
- `MAT_expert_training/results/results_notes.md`: Final Fixed/Varying expert
  selection, threshold crossing, deterministic test metrics, and published
  checkpoint identities.
- `VaryingIC_Corpus/VaryingICCorpus.jl`: Persistent generation, visualization,
  and sampling of independently seeded Rayleigh--Bénard basis snapshots for
  varying-initial-condition experiments.
- `VaryingIC_Corpus/varying_ic_corpus.jld2`: Generated corpus data.
  This file is created only by explicit generation or save calls.
- `VaryingIC_Corpus/plots/`: Generated split overview images.
- `Expert_Apprentice_Distillation/Expert_Apprentice.jl`: Shared revision copy
  of the MAT expert-apprentice implementation used as the Package-6/7/8
  training starting point. Future Fixed and Varying apprentice regressions use
  learning rate `2e-4`; their default regularized budgets are 35,000 and 50,000
  updates respectively. Proximal pruning restores groups from the current step
  with the training RNG and keeps at least one active group by default.
- `Expert_Apprentice_Distillation/DistillationCorpus.jl`: Directly includable
  Fixed-/Varying-IC teacher-rollout shard generation, atomic persistence,
  expert discovery, in-memory split merge, and lossless global-to-local MAT
  observation reconstruction.
- `Expert_Apprentice_Distillation/run_corpus_worker.jl` and
  `launch_tmux.sh`: One-shard worker and persistent parallel launcher for the
  shared distillation corpus.
- `Expert_Apprentice_Distillation/README.md`: Corpus layout, expert resolution,
  worker counts, launch, restart, and storage notes.
- `GO_Sensitivity/run_fixed_pilot.jl`: Plain-Julia local Package-6 Fixed-IC GO
  pilot with strict corpus/expert provenance checks and a scoped Pareto
  archive. It requires neither Bash nor tmux.
- `GO_Sensitivity/run_strength_calibration_pilot.jl`: Plain-Julia Package-6
  one-seed calibration orchestrator across all 22 frozen Fixed/Varying and
  grouped/separate strength cases. Every strength runs in a fresh Julia
  process; Fixed uses 35,000 updates, Varying 50,000, and all runs use
  regression learning rate `2e-4`. Run identities include phase, learning
  rate, and budget.
- `GO_Sensitivity/launch_tmux.sh`: Server launcher that submits the same 22
  cases concurrently as one self-closing tmux session per strength, with
  independent logs, resumable workers, and systemd inhibition by default.
- `GO_Sensitivity/Package6Study.jl` and `Package6Analysis.jl`: Frozen SC-only
  Package-6 production grid, deterministic paired seed plan, short job paths,
  atomic status helpers, and pure stability/selection metrics.
- `GO_Sensitivity/run_study_worker.jl`: One resumable native-SC GO or GR
  production run with strict expert/corpus provenance, configuration
  fingerprint, initial apprentice hash, evaluation shards, and explicit
  running/complete/failed status.
- `GO_Sensitivity/launch_study_tmux.sh`: Production launcher for 36 training
  sessions plus two concurrent analysis/wait sessions. It writes per-launch
  logs and machine-readable manifests and supports protocol restriction,
  preview, analysis-only restarts, and result-root override.
- `GO_Sensitivity/analyze_study_worker.jl`, `run_study_test_worker.jl`, and
  `run_study_test_episode_worker.jl`: Per-protocol completion audit, full offline
  stability analysis, compact/static and interactive plots, immutable
  validation-only GO candidate freeze, selection-inert terminal test rollouts,
  optional process-isolated per-episode parallel execution, resumable episode
  caches/statuses, CSV/JLD2 metrics, and English report generation.
- `GO_Sensitivity/test_package6_*.jl`: Synthetic manifest, pairing, metric,
  wait-state, timeout, candidate-freeze, persistence, and plot smoke tests for
  the production pipeline.
- `GO_Sensitivity/inspect_strength_calibration_pilot.jl`: PlotlyJS inspector
  that accepts only complete homogeneous long-budget blocks and exports
  four strength-colored, point-only active-group versus
  autoregressive-validation-MSE plots and highlights each combination's pooled
  nondominated checkpoints. Its default mode then launches the closed-loop
  calibration-test workers; `closed_loop_test=false` keeps inspection strictly
  on validation data.
- `GO_Sensitivity/inspect_strength_calibration_test_worker.jl`: Internal fresh-
  process worker that evaluates every retained per-strength calibration Pareto
  candidate plus the MAT expert for one protocol/grouping combination. Fixed
  uses its one shared episode; Varying uses all eight predeclared test episodes.
  Relocated results fall back from an unavailable recorded server expert path
  to the protocol-specific local expert only after enforcing the recorded
  SHA-256 expert identity.
  Episode caches, reward-curve plots, return box plots, CSV, and aggregate JLD2
  results are explicitly marked as calibration test diagnostics that must not
  drive scientific candidate selection.
- `GO_Sensitivity/inspect_fixed_pilot.jl`: Lightweight result inspector that
  prints the native GO trajectory and archived Pareto points and exports CSV
  summaries plus a PlotlyJS active-groups/validation-MSE plot. Its default
  mode additionally performs or loads a cached deterministic 200-step
  Fixed-IC closed-loop expert/native-apprentice comparison, including reward,
  full-state global Nusselt number, and action histories. Model-free inspection
  remains available through `closed_loop=false`.
- `GO_Sensitivity/README.md`: Windows-local pilot invocation, fixed pilot
  configuration, restart behavior, result inspection, and result layout.
- `Noise_Study/NoiseStudy.jl` and `prepare_manifest.jl`: Package-10 constants,
  paired noise seeds, validation-only sparse-SC and Package-6 `C_match`
  resolution, exact protocol-specific physical-channel scales, frozen
  manifests, and restart-safe result paths.
- `Noise_Study/run_worker.jl` and `launch_tmux.sh`: One worker per protocol,
  controller, and noise level; clean baseline import; sequential complete-grid
  noisy rollouts; atomic per-episode persistence; and a filtered persistent
  30-session tmux launcher without an analysis worker.
- `Noise_Study/README.md`: Package-10 protocol, server commands, restart
  behavior, output layout, and validation commands.

## Revision Run Files

The four package-2 entry points initialize their environment, policy, and hook
when included. They never start training automatically; callers invoke
`train(...)` explicitly.

The run seed is read from `REVISION_RUN_SEED` when set; otherwise manual runs
draw a random seed from `0:99_999`, matching the source scripts. Output goes to
a run-specific directory under `Run_Files/outputs` unless
`REVISION_RUN_DIRECTORY` is set before inclusion.
The retained JLD2 `save` and `load` functions persist the agent and hook in the
same style as the source run files. Higher-level experiment files own extended
checkpoint schedules, training-curve export, runtime measurement, and metadata
aggregation.

Both IPPO entry points use one shared actor/critic pair across all actuator
columns. This is parameter-sharing IPPO, not one independently parameterized
policy per actuator.

The Varying-IC entry points include `VaryingICCorpus.jl` directly and must use
`sample_initial_condition` in `generate_random_init`. The hook-compatible
zero-argument call samples from the mutable `initial_condition_split`, which
defaults to `:train`. Higher-level experiment files may change that global or
pass an explicit split, RNG, basis seed, mirror choice, and offset.
`generate_random_init` returns
`(result, split, base_seed, mirror, offset)`; the hook adapter consumes only
`result`.
Run files must never generate or mutate corpus entries.

## MAT Stability Experiment

`MAT_Stability/prepare_runs.jl` freezes five deterministic run/IC seed pairs and
the complete Varying-IC selection sequence in `results/run_plan.jld2` before
launch. `MAT_Stability/run_worker.jl` owns one
protocol/replicate/configuration. The three configuration workers in a
replicate load the same policy seed and explicit IC sequence from that plan.
The fixed and varying budgets are exactly 2,000 and 4,000 completed episodes
per configuration, respectively.

The package must preserve the paired-design checks: every worker records the
initial hashes/probes and the collector verifies that common parameter arrays,
policy RNG probes, and IC probes are equal across all three configurations;
the complete initial networks of `modified_half` and `modified_full` are
byte-identical; Varying-IC selection traces are equal within a replicate.
Separate value-chain parameters must match the main chain initially without
sharing mutable arrays.

Each completed configuration is written atomically to a descriptive JLD2 path.
Per-result lock directories prevent duplicate workers, and matching complete
files are omitted from later manifests. Failed configurations retain their
error and partial histories. Production launch uses 15 detached tmux sessions
per selected protocol so workers survive SSH disconnects.

`MAT_Stability/launch_tmux.sh` must support selecting all workers, only Fixed
IC, or only Varying IC. `MAT_Stability/collect_results.jl` collects whichever
complete, valid files are present; missing protocols, replicates, and
configurations are normal. Present failed or invalid files are reported and
ignored, while pairing mismatches among collected files remain errors.
Generated result trees are ignored by Git.

## Varying-IC Corpus Design

The corpus implementation is a directly included Julia file rather than a
module.
It is intended to be included by later revision runners:

```julia
include("Revision/VaryingIC_Corpus/VaryingICCorpus.jl")
```

Including the file defines its constants and functions directly in the
including namespace.
It loads `varying_ic_corpus.jld2` into the global mutable
`CORPUS` dictionary when the file exists, or initializes empty `:train`,
`:validation`, and `:test` dictionaries otherwise.
Including the file must never start a simulation or generate new data.

The default corpus targets are:

- 20 independently seeded training basis snapshots.
- 1 independently seeded validation basis snapshot.
- 2 independently seeded test basis snapshots.

`generate_default_corpus!` fills only missing entries and preserves existing
snapshots.
Snapshots are keyed by their generation seed within each split.
A seed must not be reused across splits.
`delete_basis_snapshot!` removes an unwanted entry, after which
`generate_default_corpus!` can refill the split with a newly drawn seed.

Corpus data and metadata must be persisted with JLD2.
Overview plots are image artifacts and are written as PNG files.

## Simulation Compatibility

The generator follows the configuration actually used by
`randomIC/randomIC_MAT.jl`:

- `Nx = 96`, `Nz = 64`
- `Lx = 2π`, `Lz = 2`
- periodic horizontal and bounded vertical topology
- `Ra = 1e4`, `Pr = 0.7`
- `UpwindBiasedFifthOrder` advection
- `RungeKutta3` time stepping
- scalar diffusivity with `ν = sqrt(Pr / Ra)` and
  `κ = 1 / sqrt(Pr * Ra)`
- uniform-grid inner time step `0.03`
- spin-up time `300`
- no-slip velocity walls
- fixed buoyancy/temperature values `2` at the bottom and `1` at the top
- independently seeded Gaussian initial perturbations with amplitude `0.2`

The older `RBtest/create_checkpoint.jl` uses `Pr = 0.71`.
Do not copy that value into revision experiments, because the trained
varying-IC environment uses `Pr = 0.7`.
The older checkpoint generator uses equal actuator actions, which reduce to the
same constant bottom value `2` used directly by the new corpus generator.

Each stored snapshot contains the interior `u`, `w`, and `b` fields as
`Float32` arrays, together with its seed, final simulation time, iteration,
creation time, and schema version.

## Initial-Condition Sampling

`sample_initial_condition` accepts a split identifier and an optional RNG,
mirror flag, and horizontal offset.
The RNG defaults to `Random.default_rng()` so callers can either use the global
RNG or pass a seeded RNG such as `StableRNG`.
Missing mirror and offset choices are drawn from the supplied RNG.
Basis seeds are sampled from a sorted key list so the same corpus and RNG state
produce the same selection.

The transformation order is horizontal reflection followed by periodic shift.
Under reflection, `b` and vertical velocity `w` are reversed along the
horizontal dimension.
Horizontal velocity `u` is face-centered on the staggered grid.
It is reversed, shifted periodically by one index to preserve face alignment,
and changes sign.
Offsets are normalized to `0:95`.

The returned named tuple contains transformed `u`, `w`, and `b` fields plus
the split, basis seed, mirror choice, and offset for reproducibility.

## Visualization

Generation functions refresh three overview images through
`plot_corpus_overviews`:

- one training overview,
- one validation overview,
- one test overview.

The default mini-plots show the `b` field.
At most five snapshots are placed in each row, so the default 20-snapshot
training overview uses a `4 × 5` layout.
All panels use common color limits for the selected field.

## Expert–Apprentice Distillation Corpus

The Package-6/7/8 distillation corpus is stored as independent atomic worker
JLD2 files. A Varying-IC worker owns exactly one
`(split, basis seed, mirror)` combination. Training workers evaluate all 96
horizontal offsets, while validation and test workers evaluate only offsets 0
and 20, always for 200 deterministic expert control steps. The default
training plan therefore has 40 workers. Validation has two workers/four
episodes and test has four workers/eight episodes. Fixed IC uses one worker and
one 200-step episode, exposed as the same in-memory training, validation, and
test dataset.

Worker files store the unique `3 × 48 × 8` global sensor tensor and `1 × 12`
expert action means as `Float32`. They do not persist the twelve overlapping
local MAT windows. `local_mat_observation` reconstructs the exact `360 × 12`
input and every production worker must pass the bit-exact reconstruction check
against the corresponding Revision MAT run file before writing data.

Including `DistillationCorpus.jl` normally scans all available complete worker
JLD2 files and merges them by protocol and split into
`DISTILLATION_CORPUS`. Generation workers explicitly disable this top-level
autoload so a new shard does not load the already generated multi-gigabyte
training set.

Production generation requires a persisted Fixed-IC or Varying-IC expert.
Every shard stores the checkpoint SHA-256 identifier, and the loader rejects
shards from different experts within one dataset. A freshly initialized MAT is
permitted only behind an explicit smoke-test flag.

## Sensor-Noise Study

Package 10 compares exactly the dense expert, the validation-only sparsest SC
Package-7/8 apprentice, and the Package-6 `C_match` apprentice. Across the four
frozen SC candidates, sparse selection first minimizes `active_inputs` and then
validation MSE among ties. This resolves to Fixed `go-sc` and Varying `gr-sc`.
Noise results must never change these selections.

The only noise model is zero-mean iid Gaussian measurement noise at levels
`0.0/0.01/0.05/0.10/0.20`. Nonzero levels use ten paired replicates per test
case. Fixed owns one case and Varying the existing eight cases. The noise seed
must not contain the controller identity.

Protocol-specific `b/w/u` scales are exact sample standard deviations from the
corresponding distillation training corpus. Position encoding is removed before
scale estimation, remains noiseless during rollouts, and is added only after
one unique physical `3 × 48 × 8` noise tensor has been generated. Local MAT
windows are reconstructed afterwards, and sparse masks are applied last.

Each worker owns one `(protocol, controller, noise level)` combination. Clean
workers import existing frozen baselines once; noisy workers run all ten
replicates and every protocol test case sequentially. Per-episode files and
worker status/results are atomic and restart-validated. The initial launcher
starts no analysis worker.

## Maintenance Rules

- Keep `Implementation_Plan.md` synchronized with revision-scope decisions and
  newly completed implementation work.
- Keep generation explicit. Never add a top-level call that fills or mutates the
  corpus.
- Keep the corpus implementation directly includable without wrapping it in a
  Julia module or requiring a subsequent `using` statement.
- Preserve split isolation at the basis-seed level.
- Preserve seeded behavior by routing all random selection through caller
  supplied RNG objects.
- Preserve the one-worker-per-`(split, basis seed, mirror)` distillation
  ownership, all-96-offset production coverage, deterministic expert means,
  atomic worker paths, and strict expert-identity checks.
- Keep compact distillation observations losslessly reconstructable as the
  exact local MAT inputs; any featurization or positional-encoding change must
  update and re-run the reconstruction audit.
- Preserve the package-3 configuration names, flags, paired seed plan, exact
  episode budgets, atomic per-configuration saves, and restart-safe paths.
- Preserve Package-10 validation-only controller selection, the Fixed `go-sc`
  and Varying `gr-sc` sparse resolution unless upstream frozen validation
  artifacts legitimately change, Package-6 `C_match` provenance, controller-
  independent paired noise seeds, protocol-specific physical channel scales,
  noiseless position encoding, global-before-local noise injection, ten
  nonzero-level replicates, and one-worker-per-combination ownership.
- Do not treat offsets or reflections of one basis snapshot as independent
  basis snapshots.
- Keep the stored field orientation compatible with `set!(model, u=..., w=...,
  b=...)` in Oceananigans.
- Update this file when the corpus schema, physical configuration, split
  protocol, transformation rules, or public API changes.
