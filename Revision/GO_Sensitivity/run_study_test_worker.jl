using Dates
using JLD2
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "Package6Study.jl"))
using .Package6Study

const TEST_STEPS = 200
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DISTILLATION_DIRECTORY = joinpath(PROJECT_ROOT, "Revision", "Expert_Apprentice_Distillation")

function ensure_test_plotly_loaded!()
    isdefined(@__MODULE__, :PlotlyJS) || Base.eval(@__MODULE__, :(using PlotlyJS))
    return nothing
end

function parse_arguments(arguments)
    values = Dict("protocol" => nothing, "results_dir" => joinpath(@__DIR__, "results", "study"), "manifest" => nothing)
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") || error("Unknown argument '$argument'.")
        index == length(arguments) && error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    isnothing(values["protocol"]) && error("--protocol is required.")
    protocol = normalize_protocol(values["protocol"])
    results_root = abspath(values["results_dir"])
    manifest = isnothing(values["manifest"]) ? joinpath(results_root, string(protocol), "analysis", "candidate_manifest.jld2") : abspath(values["manifest"])
    return (; protocol, results_root, manifest)
end

file_hash(path) = bytes2hex(SHA.sha256(read(path)))
cache_tag(material) = bytes2hex(SHA.sha256(codeunits(material)))[1:20]

function case_identifier(case)
    case === nothing && return "fixed_shared"
    return "test_b$(case.base_seed)_m$(case.mirror ? 1 : 0)_o$(case.offset)"
end

function test_cases(protocol)
    protocol === :fixed && return [nothing]
    split = CORPUS[:test]
    base_seeds = sort!(Int.(collect(keys(split))))
    length(base_seeds) == 2 || error("Expected two Varying-IC test bases, found $(length(base_seeds)).")
    return vec([(split = :test, base_seed, mirror, offset) for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)])
end

function initialize_case!(protocol, case)
    RL.reset!(env)
    if protocol === :fixed
        generate_random_init()
    else
        generate_random_init(; split = case.split, base_seed = case.base_seed, mirror = case.mirror, offset = case.offset)
    end
end

function normalize_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 && (values = dropdims(values; dims = 3))
    length(values) == 12 || error("Expected twelve actuator actions, got $(size(values)).")
    return vec(values)
end

function run_episode(protocol, case, action_function)
    initialize_case!(protocol, case)
    rewards = Vector{Float64}(undef, TEST_STEPS)
    nusselt = Vector{Float64}(undef, TEST_STEPS)
    actions = Matrix{Float32}(undef, TEST_STEPS, 12)
    for step in 1:TEST_STEPS
        action = action_function()
        actions[step, :] .= normalize_action(action)
        env(action)
        rewards[step] = mean(Float64.(env.reward))
        nusselt[step] = Float64(state_Nu(env))
        isfinite(rewards[step]) && isfinite(nusselt[step]) || error("Non-finite test result at step $step.")
    end
    return (; rewards, global_nusselt = nusselt, actions)
end

function cache_path(output, controller_id, expert_identifier, case)
    material = "$(P6_SCHEMA_VERSION)|$(controller_id)|$(expert_identifier)|$(case_identifier(case))|$TEST_STEPS"
    return joinpath(output, "cache", "$(cache_tag(material)).jld2")
end

function load_cache(path, controller_id, expert_identifier, case)
    isfile(path) || return nothing
    loaded = JLD2.load(path)
    string(loaded["controller_id"]) == controller_id || return nothing
    string(loaded["expert_identifier"]) == expert_identifier || return nothing
    string(loaded["case_identifier"]) == case_identifier(case) || return nothing
    Int(loaded["steps"]) == TEST_STEPS || return nothing
    return (rewards = Float64.(loaded["rewards"]), global_nusselt = Float64.(loaded["global_nusselt"]), actions = Float32.(loaded["actions"]))
end

function save_cache(path, episode; controller_id, expert_identifier, protocol, case)
    atomic_save(
        path;
        schema_version = P6_SCHEMA_VERSION,
        experiment = :package6_terminal_test,
        selection_influence = false,
        protocol,
        controller_id,
        expert_identifier,
        case_identifier = case_identifier(case),
        data_split = protocol === :fixed ? :shared : :test,
        steps = TEST_STEPS,
        rewards = episode.rewards,
        global_nusselt = episode.global_nusselt,
        actions = episode.actions,
        created_at = string(Dates.now()),
    )
    return episode
end

function local_expert_path(protocol, recorded)
    isfile(recorded) && return abspath(recorded)
    local_path = joinpath(DISTILLATION_DIRECTORY, "experts", string(protocol), "agent.jld2")
    isfile(local_path) || error("Expert is missing at '$recorded' and '$local_path'.")
    println("Recorded expert path is unavailable; using verified local checkpoint $local_path")
    return abspath(local_path)
end

function candidate_checkpoint(candidate, results_root, protocol)
    recorded = get(candidate, :model_path, nothing)
    !isnothing(recorded) && isfile(string(recorded)) && return abspath(string(recorded))
    isnothing(recorded) && error("Candidate $(candidate[:candidate_id]) has no model path.")
    filename = basename(replace(string(recorded), '\\' => '/'))
    recorded_run_fallback = joinpath(string(candidate[:source_run_directory]), "candidates", filename)
    if isfile(recorded_run_fallback)
        return abspath(recorded_run_fallback)
    end
    method = Symbol(candidate[:method])
    strength_index = Int(candidate[:strength_index])
    replicate = Int(candidate[:replicate])
    local_job = job_for(protocol, method, strength_index, replicate)
    relocated_fallback = joinpath(run_directory(results_root, local_job), "candidates", filename)
    isfile(relocated_fallback) || error(
        "Candidate checkpoint is missing at '$recorded', '$recorded_run_fallback', and '$relocated_fallback'.",
    )
    println("  Recorded candidate path is unavailable; using relocated checkpoint $relocated_fallback")
    return abspath(relocated_fallback)
end

function evaluate_controller!(output, results_root, protocol, cases, expert_identifier; candidate = nothing)
    controller_id = isnothing(candidate) ? "expert" : string(candidate[:candidate_id])
    candidate_model = nothing
    input_mask = nothing
    if !isnothing(candidate)
        loaded = JLD2.load(candidate_checkpoint(candidate, results_root, protocol))
        candidate_model = loaded["model_payload"]
        Flux.testmode!(candidate_model)
        input_mask = Float32.(candidate[:mask])
    end
    episodes = Dict{String, Any}()
    for case in cases
        path = cache_path(output, controller_id, expert_identifier, case)
        episode = load_cache(path, controller_id, expert_identifier, case)
        if isnothing(episode)
            println("  $controller_id: $(case_identifier(case))")
            action_function = if isnothing(candidate)
                () -> RL.prob(agent.policy, env).μ
            else
                () -> RL.prob(candidate_model, env.state .* input_mask, nothing).μ[:, :, 1]
            end
            episode = run_episode(protocol, case, action_function)
            save_cache(path, episode; controller_id, expert_identifier, protocol, case)
        end
        episodes[case_identifier(case)] = episode
    end
    return episodes
end

episode_matrix(episodes, cases, field) = reduce(vcat, [permutedims(Float64.(episodes[case_identifier(case)][field])) for case in cases])

function write_test_csv(path, cases, controllers)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "controller,role,case,step,reward,global_nusselt")
        for controller in controllers, case in cases
            episode = controller.episodes[case_identifier(case)]
            for step in 1:TEST_STEPS
                println(io, join((controller.id, controller.role, case_identifier(case), step, episode.rewards[step], episode.global_nusselt[step]), ','))
            end
        end
    end
end

function write_return_csv(path, cases, controllers)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "controller,role,case,return")
        for controller in controllers, case in cases
            episode = controller.episodes[case_identifier(case)]
            println(io, join((controller.id, controller.role, case_identifier(case), sum(episode.rewards)), ','))
        end
    end
    return path
end

function plot_rewards(output, protocol, cases, controllers)
    ensure_test_plotly_loaded!()
    return Base.invokelatest(plot_rewards_loaded, output, protocol, cases, controllers)
end

function plot_rewards_loaded(output, protocol, cases, controllers)
    colors = ("#202020", "#2166AC", "#B2182B")
    traces = PlotlyJS.GenericTrace[]
    for (index, controller) in enumerate(controllers)
        matrix = episode_matrix(controller.episodes, cases, :rewards)
        push!(traces, scatter(
            x = collect(1:TEST_STEPS), y = vec(mean(matrix; dims = 1)), mode = "lines",
            name = controller.label, line = attr(color = colors[mod1(index, length(colors))], width = index == 1 ? 3.5 : 2.7),
        ))
    end
    plot = Plot(traces, Layout(
        template = "plotly_white", width = 900, height = 540,
        title = "$(protocol === :fixed ? "Fixed" : "Varying") terminal test reward",
        xaxis = attr(title = "Control step"), yaxis = attr(title = "Mean reward (higher is better)"),
    ))
    path = joinpath(output, "test_reward_curves.svg")
    PlotlyJS.savefig(plot, path; width = 900, height = 540)
    return path
end

function plot_returns(output, cases, controllers)
    ensure_test_plotly_loaded!()
    return Base.invokelatest(plot_returns_loaded, output, cases, controllers)
end

function plot_returns_loaded(output, cases, controllers)
    traces = PlotlyJS.GenericTrace[]
    for controller in controllers
        matrix = episode_matrix(controller.episodes, cases, :rewards)
        push!(traces, box(y = vec(sum(matrix; dims = 2)), name = controller.label, boxpoints = "all"))
    end
    plot = Plot(traces, Layout(template = "plotly_white", width = 850, height = 540, title = "Varying-IC terminal test returns", yaxis = attr(title = "200-step return")))
    path = joinpath(output, "test_return_boxplot.svg")
    PlotlyJS.savefig(plot, path; width = 850, height = 540)
    return path
end

function run_test_worker(options)
    isfile(options.manifest) || error("Frozen candidate manifest is missing: $(options.manifest)")
    manifest_hash_before = file_hash(options.manifest)
    manifest = JLD2.load(options.manifest)
    Bool(manifest["frozen_before_test"]) || error("Candidate manifest is not marked frozen.")
    Symbol(manifest["protocol"]) === options.protocol || error("Candidate manifest protocol mismatch.")
    candidates = [Dict{Symbol, Any}(Symbol(key) => value for (key, value) in raw) for raw in manifest["candidates"]]
    1 <= length(candidates) <= 2 || error("Expected one or two selected GO candidates.")
    expert_identifier = string(manifest["expert_identifier"])
    expert_path = local_expert_path(options.protocol, string(manifest["expert_path"]))
    output = joinpath(options.results_root, string(options.protocol), "analysis", "test")
    ENV["DISTILLATION_PROTOCOL"] = string(options.protocol)
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    ENV["DISTILLATION_GROUP_CHANNELS"] = "false"
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV[options.protocol === :fixed ? "DISTILLATION_FIXED_EXPERT_PATH" : "DISTILLATION_VARYING_EXPERT_PATH"] = expert_path
    ENV["REVISION_RUN_SEED"] = string(P6_MASTER_SEED)
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output, "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output, "apprentice_output")
    include(joinpath(DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
    loaded_identifier = string(DISTILLATION_EXPERT_METADATA[:identifier])
    loaded_identifier == expert_identifier || error("Local expert $loaded_identifier does not match study expert $expert_identifier.")
    cases = test_cases(options.protocol)
    expert_episodes = evaluate_controller!(output, options.results_root, options.protocol, cases, expert_identifier)
    controllers = NamedTuple[(id = "expert", role = "expert", label = "MAT expert", episodes = expert_episodes, candidate = nothing)]
    for candidate in candidates
        episodes = evaluate_controller!(output, options.results_root, options.protocol, cases, expert_identifier; candidate)
        role = string(candidate[:selection_role])
        label = @sprintf("%s: λ=%.4g, %d groups", role, Float64(candidate[:regularization_strength]), Int(candidate[:active_groups]))
        push!(controllers, (id = string(candidate[:candidate_id]), role, label, episodes, candidate))
    end
    file_hash(options.manifest) == manifest_hash_before || error("Candidate manifest changed during test execution.")
    csv_path = joinpath(output, "test_episodes.csv")
    write_test_csv(csv_path, cases, controllers)
    return_csv_path = write_return_csv(joinpath(output, "test_returns.csv"), cases, controllers)
    reward_plot = plot_rewards(output, options.protocol, cases, controllers)
    return_plot = options.protocol === :varying ? plot_returns(output, cases, controllers) : nothing
    summaries = [begin
        matrix = episode_matrix(controller.episodes, cases, :rewards)
        (id = controller.id, role = controller.role, mean_return = mean(vec(sum(matrix; dims = 2))), returns = vec(sum(matrix; dims = 2)))
    end for controller in controllers]
    result_path = joinpath(output, "test_results.jld2")
    atomic_save(
        result_path;
        schema_version = P6_SCHEMA_VERSION,
        experiment = :package6_terminal_test,
        protocol = options.protocol,
        selection_influence = false,
        candidate_manifest = options.manifest,
        candidate_manifest_sha256 = manifest_hash_before,
        cases,
        summaries,
        csv_path,
        return_csv_path,
        reward_plot,
        return_plot,
        completed_at = string(Dates.now()),
    )
    println("Terminal test complete: $result_path")
    return result_path
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        run_test_worker(parse_arguments(ARGS))
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
