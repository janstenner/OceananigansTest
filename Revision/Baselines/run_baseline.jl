using Dates
using JLD2
using SHA
using Statistics
using UUIDs

const BASELINE_SCHEMA_VERSION = 1
const BASELINE_STEPS = 200
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DISTILLATION_ROOT = joinpath(PROJECT_ROOT, "Revision", "Expert_Apprentice_Distillation")
const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "results")
const DEFAULT_RUN_SEED = 600_600

function usage()
    println("""
    Usage: julia --startup-file=no --project=. Revision/Baselines/run_baseline.jl [options]

      --protocol fixed|varying
      --controller expert|unactuated
      --results-dir PATH
      --expert-path PATH
      --run-seed N
      --overwrite
      --help
    """)
end

function parse_options(arguments)
    values = Dict{String, Any}(
        "protocol" => nothing,
        "controller" => nothing,
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "expert_path" => nothing,
        "run_seed" => DEFAULT_RUN_SEED,
        "overwrite" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--overwrite"
            values["overwrite"] = true
            index += 1
            continue
        end
        startswith(argument, "--") || error("Unknown argument '$argument'.")
        index < length(arguments) || error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    isnothing(values["protocol"]) && error("--protocol is required.")
    isnothing(values["controller"]) && error("--controller is required.")
    protocol = Symbol(lowercase(string(values["protocol"])))
    controller = Symbol(lowercase(string(values["controller"])))
    protocol in (:fixed, :varying) || error("Protocol must be fixed or varying.")
    controller in (:expert, :unactuated) || error("Controller must be expert or unactuated.")
    run_seed = parse(Int, string(values["run_seed"]))
    run_seed >= 0 || error("--run-seed must be nonnegative.")
    default_expert = joinpath(DISTILLATION_ROOT, "experts", string(protocol), "agent.jld2")
    expert_path = abspath(isnothing(values["expert_path"]) ? default_expert : string(values["expert_path"]))
    return (;
        protocol,
        controller,
        results_root = abspath(string(values["results_dir"])),
        expert_path,
        run_seed,
        overwrite = Bool(values["overwrite"]),
    )
end

baseline_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

run_file(options) = joinpath(
    PROJECT_ROOT,
    "Revision",
    "Run_Files",
    options.protocol === :fixed ? "FixedIC_MAT.jl" : "VaryingIC_MAT.jl",
)

test_data_file(options) = options.protocol === :fixed ?
    joinpath(PROJECT_ROOT, "RBmodel300.jld2") :
    joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus.jld2")

result_path(options) = joinpath(
    options.results_root,
    string(options.protocol),
    "$(options.controller).jld2",
)

function provenance(options)
    runtime_path = run_file(options)
    data_path = test_data_file(options)
    isfile(runtime_path) || error("Run file is missing: $runtime_path")
    isfile(data_path) || error("Test data are missing: $data_path")
    if options.controller === :expert
        isfile(options.expert_path) || error("Expert checkpoint is missing: $(options.expert_path)")
    end
    return (
        run_file_path = runtime_path,
        run_file_sha256 = baseline_sha256(runtime_path),
        test_data_path = data_path,
        test_data_sha256 = baseline_sha256(data_path),
        expert_path = options.controller === :expert ? options.expert_path : "",
        expert_sha256 = options.controller === :expert ? baseline_sha256(options.expert_path) : "",
    )
end

function result_is_current(path, options, source)
    isfile(path) || return false
    try
        loaded = JLD2.load(path)
        return Int(loaded["schema_version"]) == BASELINE_SCHEMA_VERSION &&
               string(loaded["status"]) == "complete" &&
               Symbol(loaded["protocol"]) === options.protocol &&
               Symbol(loaded["controller"]) === options.controller &&
               Int(loaded["run_seed"]) == options.run_seed &&
               string(loaded["run_file_sha256"]) == source.run_file_sha256 &&
               string(loaded["test_data_sha256"]) == source.test_data_sha256 &&
               string(loaded["expert_sha256"]) == source.expert_sha256
    catch
        return false
    end
end

latest_runtime_binding(name::Symbol) = Base.invokelatest(() -> getfield(@__MODULE__, name))

function configure_runtime!(options, source, runtime_directory)
    ENV["REVISION_RUN_SEED"] = string(options.run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    include(source.run_file_path)
    if options.controller === :expert
        include(joinpath(DISTILLATION_ROOT, "DistillationCorpus.jl"))
        loader = latest_runtime_binding(:load_distillation_expert!)
        metadata = Base.invokelatest(loader, options.protocol; explicit_path = options.expert_path)
        string(metadata[:checkpoint_sha256]) == source.expert_sha256 || error(
            "Loaded expert identity differs from the requested checkpoint.",
        )
    end
    return nothing
end

function test_cases(protocol)
    protocol === :fixed && return [(case_id = "fixed_shared", choice = nothing)]
    corpus = latest_runtime_binding(:CORPUS)
    base_seeds = sort!(Int.(collect(keys(corpus[:test]))))
    length(base_seeds) == 2 || error("Expected two Varying test bases, found $(length(base_seeds)).")
    return vec([
        (
            case_id = "base_$(base_seed)_mirror_$(Int(mirror))_offset_$(offset)",
            choice = (split = :test, base_seed, mirror, offset),
        )
        for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)
    ])
end

function reset_episode!(protocol, choice)
    initialize = latest_runtime_binding(:generate_random_init)
    if protocol === :fixed
        Base.invokelatest(initialize)
    else
        Base.invokelatest(
            initialize;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        )
    end
    runtime_rl = latest_runtime_binding(:RL)
    runtime_env = latest_runtime_binding(:env)
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    return nothing
end

function controller_action(controller)
    controller === :unactuated && return zeros(Float32, 1, 12)
    runtime_rl = latest_runtime_binding(:RL)
    runtime_agent = latest_runtime_binding(:agent)
    runtime_env = latest_runtime_binding(:env)
    action = Base.invokelatest(runtime_rl.prob, runtime_agent.policy, runtime_env).μ
    hasproperty(runtime_agent.policy, :clip1) && runtime_agent.policy.clip1 && clamp!(action, -1.0f0, 1.0f0)
    return action
end

function run_episode(options, case)
    reset_episode!(options.protocol, case.choice)
    runtime_env = latest_runtime_binding(:env)
    nusselt_function = latest_runtime_binding(:state_Nu)
    rewards = Vector{Float64}(undef, BASELINE_STEPS)
    state_nusselt = Vector{Float64}(undef, BASELINE_STEPS)
    actions = Matrix{Float32}(undef, BASELINE_STEPS, 12)
    for step in 1:BASELINE_STEPS
        action = Base.invokelatest(controller_action, options.controller)
        action_values = vec(Float32.(Array(action)))
        length(action_values) == 12 || error("Expected 12 actions, got $(length(action_values)).")
        actions[step, :] .= action_values
        Base.invokelatest(runtime_env, action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        state_nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
    end
    all(isfinite, rewards) && all(isfinite, state_nusselt) || error(
        "Non-finite reward or state_Nu in $(case.case_id).",
    )
    return (;
        case_id = case.case_id,
        choice = case.choice,
        rewards,
        state_nusselt,
        actions,
        reward_sum = sum(rewards),
        sum_state_nusselt = sum(state_nusselt),
        negative_sum_state_nusselt = -sum(state_nusselt),
        mean_state_nusselt = mean(state_nusselt),
    )
end

function atomic_save(path; kwargs...)
    mkpath(dirname(path))
    temporary = joinpath(dirname(path), ".$(basename(path)).$(getpid()).$(uuid4()).tmp")
    try
        JLD2.jldsave(temporary; kwargs...)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    source = provenance(options)
    output = result_path(options)
    if !options.overwrite && result_is_current(output, options, source)
        println("Current baseline already exists: $output")
        return output
    end
    mktempdir(; prefix = "oceananigans-baseline-") do runtime_directory
        configure_runtime!(options, source, runtime_directory)
        cases = test_cases(options.protocol)
        episodes = map(cases) do case
            println("Running $(options.protocol)/$(options.controller): $(case.case_id)")
            episode = Base.invokelatest(run_episode, options, case)
            println(
                "  reward_sum=$(episode.reward_sum), " *
                "sum_state_Nu=$(episode.sum_state_nusselt)",
            )
            episode
        end
        atomic_save(
            output;
            schema_version = BASELINE_SCHEMA_VERSION,
            status = :complete,
            experiment = :revision_test_baseline,
            protocol = options.protocol,
            controller = options.controller,
            policy = options.controller === :expert ? :deterministic_mean_action : :zero_action,
            run_seed = options.run_seed,
            steps = BASELINE_STEPS,
            case_count = length(episodes),
            source...,
            episodes,
            completed_at = string(Dates.now(Dates.UTC)),
        )
    end
    println("Saved baseline: $output")
    return output
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
