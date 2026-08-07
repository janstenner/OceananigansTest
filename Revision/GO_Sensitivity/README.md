# Package 6 — GO Sensitivity

This directory owns the Package-6 experiment orchestration. Shared apprentice
training, distillation-corpus loading, mask generation, and Pareto storage stay
in `Revision/Expert_Apprentice_Distillation`.

## Fixed-IC technical pilot

`run_fixed_pilot.jl` runs one explicitly non-scientific local pilot:

- Fixed IC
- GO with `regularization_strength = 0.09`
- apprentice seed `600601`
- 500 optimizer/proximal updates
- batch size 20
- candidate evaluation every 25 updates starting at update 0
- autoregressive validation on the shared 200-sample Fixed corpus
- native sparsity plus two `:pilot_only` hard-threshold checks
- no post-pruning finetuning

It requires these existing files:

```text
Revision/Expert_Apprentice_Distillation/experts/fixed/agent.jld2
Revision/Expert_Apprentice_Distillation/worker_results/fixed/fixed_shared.jld2
```

The runner is plain Julia. It does not use Bash, tmux, or server-specific
process handling. From Windows PowerShell in the project root:

```powershell
julia --project=. .\Revision\GO_Sensitivity\run_fixed_pilot.jl
```

Alternatively, from a fresh Julia REPL started with the project activated:

```julia
include("Revision/GO_Sensitivity/run_fixed_pilot.jl")
run_fixed_pilot()
```

Results are written below `Revision/GO_Sensitivity/results/fixed_pilot/` and
ignored by Git. An interrupted run resumes from `resume/latest.jld2`. A
completed run is reported and not repeated. A numerical failure is retained
for inspection and is not silently resumed.

The pilot validates the real end-to-end path and measures local runtime. Its
thresholds and results are not Package-6 production decisions. Production
training budgets, evaluation intervals, apprentice seeds, and the complete
Package-7/8 threshold list are frozen only after inspecting this pilot.

