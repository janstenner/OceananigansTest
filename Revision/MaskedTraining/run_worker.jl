include("MaskedStudy.jl")
include(joinpath(@__DIR__, "..", "MAT_IPPO_Comparison", "MATIPPOExperiment.jl"))
include("MaskedTraining.jl")
using JLD2, Dates, Flux, Statistics, SHA
const S = MaskedStudy
const M = MATIPPOExperiment

# Package-3 imports use the same arrays but prefix their hash with the number
# of parameter arrays. Reproduce that serialization without loading an agent.
function legacy_parameter_hash(agent)
    parameters = Flux.trainables((agent.policy.encoder, agent.policy.decoder))
    io = IOBuffer()
    write(io, string(length(parameters)), UInt8(0))
    for p in parameters
        a = Array(p)
        write(io, string(eltype(a)), UInt8(0), string(size(a)), UInt8(0))
        write(io, reinterpret(UInt8, vec(a)))
    end
    bytes2hex(SHA.sha256(take!(io)))
end

function run_worker(; results, protocol, grouping, run_id, episodes = nothing)
    m = JLD2.load(S.manifest_path(results), "manifest")
    m.rayleigh == 1e4 || error("Package 11 requires Ra=1e4")
    c = only(filter(c -> c.protocol === protocol && c.grouping === grouping, m.candidates))
    e = only(filter(e -> e.run_id == run_id, m.entries))
    budget = something(episodes, S.BUDGETS[protocol])
    0 <= budget <= S.BUDGETS[protocol] || error("Invalid episode budget")
    path = S.result_path(results, protocol, grouping, run_id)
    lock = M.acquire_lock(path * ".lock")
    started = now()
    try
        S.complete_result(path, m.identity, budget) && return path
        for (relative, hash) in m.source_hashes
            S.file_sha256(joinpath(S.ROOT, relative)) == hash || error("Frozen source changed: $relative")
        end
        for (name, hash) in m.rl_sources
            S.file_sha256(joinpath(dirname(pathof(RL)), name)) == hash || error("Frozen RL source changed: $name")
        end
        baseline = m.baselines[(run_id, protocol)]
        S.file_sha256(baseline.path) == baseline.sha256 || error("Dense reference changed")
        M.include_run_file!(protocol, :mat, e.run_seed, joinpath(dirname(path), "_bootstrap_$run_id"))
        # Julia 1.12 also world-ages newly included global bindings (Ra, etc.).
        return Base.invokelatest() do
            M.configure_agent!(protocol, :mat, e)
            M.Ra == 1e4 || error("Run file is not Ra=1e4")
            length(c.mask) == size(M.env.state, 1) || error("Mask shape mismatch")
            M.parameter_count(:mat) == M.EXPECTED_PARAMETERS[(protocol, :mat)] || error("Architecture mismatch")
            for (name, value) in baseline.parameters
                isdefined(M, Symbol(name)) || continue
                getfield(M, Symbol(name)) == value || error("Dense parameter mismatch: $name")
            end
            initial_hash = M.parameter_hash(:mat)
            isnothing(baseline.initial_hash) || initial_hash == baseline.initial_hash || error("Dense initialization mismatch")
            isnothing(baseline.legacy_initial_hash) || legacy_parameter_hash(M.agent) == baseline.legacy_initial_hash || error("Imported dense initialization mismatch")
            trace = NamedTuple[]
            M.configure_training_initializers!(protocol, e, trace)
            steps = 0
            elapsed = @elapsed steps = Base.invokelatest(train_masked!, M.agent, M.hook, M.env, c.mask, budget)
            protocol === :varying && trace != e.varying_trace[1:budget] && error("Training IC trace mismatch")
            # Reuse the Comparison result schema, and publish only after adding the
            # frozen mask/seed/reference identity. No dense files are modified.
            temporary = path * ".pending.$(getpid())"
            M.save_training_result(temporary, e, protocol, :mat, budget, elapsed, started, trace, initial_hash)
            JLD2.jldopen(temporary, "a+") do f
                f["manifest_identity"] = m.identity
                f["experiment"] = :package11_direct_masked_rl
                f["candidate"] = c
                f["dense_reference"] = baseline
                f["control_steps"] = steps
                f["control_steps_per_second"] = elapsed > 0 ? steps / elapsed : 0.0
                f["reward_uses_full_sensors"] = true
            end
            mv(temporary, path; force = true)
            return path
        end
    catch err
        S.atomic_save(path * ".failure.jld2"; status = "failed", manifest_identity = m.identity,
            protocol, grouping, run_id, episode_target = budget, started_at = string(started),
            error_message = sprint(showerror, err, catch_backtrace()),
            rewards = isnothing(M.hook) ? Float64[] : copy(M.hook.rewards))
        rethrow()
    finally
        rm(lock; recursive = true, force = true)
    end
end

function main(args = ARGS)
    o = S.parse_options(args, Dict{String, Any}("results_dir" => S.DEFAULT_RESULTS,
        "protocol" => nothing, "grouping" => nothing, "run_id" => nothing, "episodes" => nothing))
    all(!isnothing(o[k]) for k in ("protocol", "grouping", "run_id")) || error("protocol, grouping and run-id are required")
    println(run_worker(; results = abspath(o["results_dir"]), protocol = Symbol(o["protocol"]),
        grouping = Symbol(o["grouping"]), run_id = o["run_id"],
        episodes = isnothing(o["episodes"]) ? nothing : parse(Int, o["episodes"])))
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
