# Optional integration smoke: one fresh process per protocol. Two physical
# control steps and one PPO update; no production result is written.
include("run_worker.jl")
using Test

protocol = Symbol(only(ARGS))
entry = first(M.load_plan(S.DEFAULT_COMPARISON)["entries"])
package = protocol === :fixed ? "Package7/results/260830_173924" : "Package8/results/260830_231109"
c = S.select_candidate(protocol, :sc, joinpath(S.ROOT, "Revision", package))
baseline = S.baseline_record(S.DEFAULT_COMPARISON, entry, protocol)
mktempdir() do tmp
    M.include_run_file!(protocol, :mat, entry.run_seed, tmp)
    Base.invokelatest() do
        M.configure_agent!(protocol, :mat, entry)
        @test M.Ra == 1e4
        @test legacy_parameter_hash(M.agent) == baseline.legacy_initial_hash
        trace = NamedTuple[]
        M.configure_training_initializers!(protocol, entry, trace)
        M.env.te = 2 * M.env.dt
        M.agent.policy.update_freq = 2
        M.agent.policy.n_microbatches = 1
        M.agent.policy.n_epochs = 1
        initial_hash = M.parameter_hash(:mat)
        @test Base.invokelatest(train_masked!, M.agent, M.hook, M.env, c.mask, 1) == 2
        @test M.parameter_hash(:mat) != initial_hash
        @test length(M.hook.rewards) == 1
        @test all(isfinite, M.hook.rewards)
        protocol === :varying && @test trace == entry.varying_trace[1:1]
        println("$protocol: full-sensor reward=$(M.hook.rewards[1]), one masked PPO update passed.")
    end
end
