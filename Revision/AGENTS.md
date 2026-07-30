# AGENTS.md

## Purpose

This file documents the revision-specific implementation under `Revision/`.
The revision code supports the new experiments for the resubmission of the
Sparse Sensing paper.

## Directory Structure

- `VaryingIC_Corpus/VaryingICCorpus.jl`: Persistent generation, visualization,
  and sampling of independently seeded Rayleigh--Bénard basis snapshots for
  varying-initial-condition experiments.
- `VaryingIC_Corpus/varying_ic_corpus.jld2`: Generated corpus data.
  This file is created only by explicit generation or save calls.
- `VaryingIC_Corpus/plots/`: Generated split overview images.

## Varying-IC Corpus Design

The corpus module is self-contained and is intended to be included by later
revision runners:

```julia
include("Revision/VaryingIC_Corpus/VaryingICCorpus.jl")
using .VaryingICCorpus
```

Including the module loads `varying_ic_corpus.jld2` into the global mutable
`CORPUS` dictionary when the file exists, or initializes empty `:train`,
`:validation`, and `:test` dictionaries otherwise.
Including the module must never start a simulation or generate new data.

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

- Keep generation explicit. Never add a top-level call that fills or mutates the
  corpus.
- Preserve split isolation at the basis-seed level.
- Preserve seeded behavior by routing all random selection through caller
  supplied RNG objects.
- Do not treat offsets or reflections of one basis snapshot as independent
  basis snapshots.
- Keep the stored field orientation compatible with `set!(model, u=..., w=...,
  b=...)` in Oceananigans.
- Update this file when the corpus schema, physical configuration, split
  protocol, transformation rules, or public API changes.
