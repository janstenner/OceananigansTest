# Package 10 — Sensor-noise study

This directory implements the post-hoc, no-retraining robustness study for the
Fixed-IC and Varying-IC controllers. Selection and all scale definitions are
frozen in protocol-specific manifests before the first noisy rollout.

## Frozen controllers

Each protocol compares exactly three controllers:

1. the published dense MAT expert;
2. the final sparse SC apprentice selected across the four Package-7/8 SC
   configurations by minimum `active_inputs`, then minimum validation MSE among
   ties;
3. the validation-only `C_match` candidate from the Package-6 GO sensitivity
   manifest.

With the current frozen artifacts this selects Package-7 `go-sc` for Fixed IC
and Package-8 `gr-sc` for Varying IC. `C_sparse` is not part of this study.
The manifest records every SC candidate considered by the sparse-selection
rule, both selection files, checkpoint hashes, masks, and expert identity.

## Noise protocol

- additive, zero-mean, independent Gaussian measurement noise;
- levels `(0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)`;
- ten noise replicates per test case at every nonzero level;
- one Fixed test case and eight Varying test cases;
- deterministic mean actions and 200 control steps;
- no retraining, clipping, bias, dropout, or temporal/spatial correlation.

The three scales are protocol-specific sample standard deviations of physical
`b`, `w`, and `u` values from the complete distillation training corpus. The
stored sinusoidal position encoding is removed before scale calculation.
During a rollout one unique `3 × 48 × 8` physical noise tensor is generated per
control step. Position encoding is added only afterwards, and the overlapping
`360 × 12` MAT windows are then reconstructed. Thus repeated appearances of
one physical sensor share one noisy value. Sparse masks are applied last, so
inactive inputs remain exactly zero.

The noise seed depends only on protocol, level, replicate, and test case—not on
the controller. Expert, sparse apprentice, and `C_match` therefore receive the
same standardized white-noise sequence for every paired run.

## Launching

Preview the complete 48-session launch without writing files:

```bash
bash Revision/Noise_Study/launch_tmux.sh --preview
```

Launch both protocols:

```bash
bash Revision/Noise_Study/launch_tmux.sh
```

Useful restricted launches include:

```bash
bash Revision/Noise_Study/launch_tmux.sh --protocol fixed
bash Revision/Noise_Study/launch_tmux.sh --protocol varying --noise-level 0.10
bash Revision/Noise_Study/launch_tmux.sh --experiment-id ID \
  --noise-level 0.30 --noise-level 0.40 --noise-level 0.50
bash Revision/Noise_Study/launch_tmux.sh --controller c_match --retry-failed
```

The launcher behaves like the other revision launchers: detached tmux sessions,
one log per session, launch metadata, active-session detection, thread limits,
filters, preview, and explicit failed-run retry. Sessions close when their
workers exit.

On its first real launch it creates one manifest per selected protocol under
`results/<experiment-id>/manifests/`. Exact Varying scale computation reads all
40 training shards and can take several minutes. Existing manifests are reused;
`--rebuild-manifest` deliberately recomputes and replaces them.
Workers also accept newly added supported noise levels when reusing an older
manifest whose recorded grid ends at `0.20`; the frozen scales and controller
selection remain unchanged.

## Worker ownership and restart

One worker owns one `(protocol, controller, noise level)` combination:

- level zero imports the already frozen clean baseline once for every test case;
- each nonzero Fixed worker runs `1 case × 10 replicates`;
- each nonzero Varying worker runs `8 cases × 10 replicates`.

Every episode is written atomically below
`results/<experiment-id>/<protocol>/<controller>/<level>/episodes/`. A restarted
worker validates and skips matching complete episodes. It writes a compact
`result.jld2` only after the complete owned grid is available. Failed workers
record their traceback in `status.jld2` and require `--retry-failed`.

There is intentionally no analysis worker in this package yet. The later
standalone analysis consumes the atomic worker summaries. The first paper-table
script is:

```bash
julia --startup-file=no --project=. Revision/Noise_Study/make_paper_tables.jl
```

Without arguments it independently chooses the newest Fixed and Varying
experiment directories. Different experiment IDs only produce warnings and are
combined. Incomplete worker cells are written as `NA`. The script creates a
wide CSV and Markdown table with the mean test-set `state_Nu` for every
controller and noise level, ordered as expert, `C_match`, then sparse, plus
compact JLD2 metrics and SHA-256 provenance.

## Validation

Run the lightweight deterministic and artifact-selection tests with:

```bash
julia --startup-file=no --project=. Revision/Noise_Study/test_noise_study.jl
bash -n Revision/Noise_Study/launch_tmux.sh
```
