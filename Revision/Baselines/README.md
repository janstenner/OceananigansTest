# Test baselines

This directory evaluates the tracked Fixed- and Varying-IC MAT experts and the
corresponding zero-action baselines on exactly the terminal test cases used by
the revision studies: one shared Fixed episode and eight Varying episodes from
the two test bases, both mirrors, and offsets 0 and 20.

Start all four independent tmux workers from the repository root:

```bash
bash Revision/Baselines/launch_tmux.sh
```

Use `--preview`, `--overwrite`, `--protocol fixed|varying`, or
`--controller expert|unactuated` to restrict or repeat the launch. Expert paths,
the result directory, run seed, and BLAS/OMP thread counts are also configurable
through the launcher's documented options.

The four atomic outputs are:

```text
results/fixed/expert.jld2
results/fixed/unactuated.jld2
results/varying/expert.jld2
results/varying/unactuated.jld2
```

Each file contains all 200-step reward, full-state `state_Nu`, and action
curves; episode-level sums and means; exact test-case choices; and SHA-256
identities for the expert, runtime file, and underlying test data. A matching
existing result is reused, while a changed expert, run file, or test-data file
causes automatic regeneration.
