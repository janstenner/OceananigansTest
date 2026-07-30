# AGENTS.md

## Purpose

This file documents the revision-specific implementation under `Revision/`.
The revision code supports the new experiments for the resubmission of the
Sparse Sensing paper.

## Directory Structure

- `Implementation_Plan.md`: Ordered implementation and experiment checklist for
  the paper revision.
- `Run_Files/FixedIC_MAT.jl` and `Run_Files/FixedIC_IPPO.jl`: Standalone
  fixed-initial-condition entry points.
- `Run_Files/VaryingIC_MAT.jl` and `Run_Files/VaryingIC_IPPO.jl`: Standalone
  varying-initial-condition entry points backed by the corpus.
- `Run_Files/README.md`: Public run-file configuration and orchestration notes.
- `MAT_Stability/`: Paired package-3 MAT workers, detached tmux launcher,
  atomic JLD2 results, and strict result collector.
- `MAT_Stability/README.md`: Server launch, restart, result, and collection
  instructions for package 3.
- `VaryingIC_Corpus/VaryingICCorpus.jl`: Persistent generation, visualization,
  and sampling of independently seeded Rayleigh--Bénard basis snapshots for
  varying-initial-condition experiments.
- `VaryingIC_Corpus/varying_ic_corpus.jld2`: Generated corpus data.
  This file is created only by explicit generation or save calls.
- `VaryingIC_Corpus/plots/`: Generated split overview images.

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

`MAT_Stability/run_worker.jl` owns one protocol/replicate and initializes the
three MAT configurations `python_like`, `modified_half`, and `modified_full`
sequentially. A deterministic seed plan supplies the same policy seed and, for
Varying IC, the same IC-sampling seed to all three configurations in a
replicate. The fixed and varying budgets are exactly 2,000 and 4,000 completed
episodes per configuration, respectively.

The worker must preserve the paired-design checks: common parameter arrays,
policy RNG probes, and IC probes are equal across all three configurations;
the complete initial networks of `modified_half` and `modified_full` are
byte-identical; Varying-IC selection traces are equal within a replicate.
Separate value-chain parameters must match the main chain initially without
sharing mutable arrays.

Each completed configuration is written atomically to a descriptive JLD2 path.
Per-replicate lock directories prevent duplicate workers, and matching complete
files are skipped on restart. Failed configurations retain their error and
partial histories. Production launch uses detached tmux sessions so workers
survive SSH disconnects.

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
- Preserve the package-3 configuration names, flags, paired seed plan, exact
  episode budgets, atomic per-configuration saves, and restart-safe paths.
- Do not treat offsets or reflections of one basis snapshot as independent
  basis snapshots.
- Keep the stored field orientation compatible with `set!(model, u=..., w=...,
  b=...)` in Oceananigans.
- Update this file when the corpus schema, physical configuration, split
  protocol, transformation rules, or public API changes.
