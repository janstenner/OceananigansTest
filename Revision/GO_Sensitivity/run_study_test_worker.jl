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
const TEST_EPISODE_WORKER = joinpath(@__DIR__, "run_study_test_episode_worker.jl")

function ensure_test_plotly_loaded!()
    isdefined(@__MODULE__, :PlotlyJS) || Base.eval(@__MODULE__, :(using PlotlyJS))
    return nothing
end

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "protocol" => nothing,
        "results_dir" => joinpath(@__DIR__, "results", "study"),
        "manifest" => nothing,
        "parallel_test" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--parallel-test"
            values["parallel_test"] = true
            index += 1
        elseif startswith(argument, "--")
            index == length(arguments) && error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(values, key) || error("Unknown option '$argument'.")
            values[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument '$argument'.")
        end
    end
    isnothing(values["protocol"]) && error("--protocol is required.")
    protocol = normalize_protocol(values["protocol"])
    results_root = abspath(values["results_dir"])
    manifest = isnothing(values["manifest"]) ? joinpath(results_root, string(protocol), "analysis", "candidate_manifest.jld2") : abspath(values["manifest"])
    return (; protocol, results_root, manifest, parallel_test = Bool(values["parallel_test"]))
end

file_hash(path) = bytes2hex(SHA.sha256(read(path)))
cache_tag(material) = bytes2hex(SHA.sha256(codeunits(material)))[1:20]

function case_identifier(case)
    case === nothing && return "fixed_shared"
    case isa AbstractString && return string(case)
    return "test_b$(case.base_seed)_m$(case.mirror ? 1 : 0)_o$(case.offset)"
end

function test_cases(protocol)
    protocol === :fixed && return [nothing]
    corpus = Base.invokelatest(() -> getfield(@__MODULE__, :CORPUS))
    split = corpus[:test]
    base_seeds = sort!(Int.(collect(keys(split))))
    length(base_seeds) == 2 || error("Expected two Varying-IC test bases, found $(length(base_seeds)).")
    return vec([(split = :test, base_seed, mirror, offset) for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)])
end

function initialize_case!(protocol, case)
    runtime_rl = getfield(@__MODULE__, :RL)
    runtime_env = getfield(@__MODULE__, :env)
    initialize_episode = getfield(@__MODULE__, :generate_random_init)
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    if protocol === :fixed
        Base.invokelatest(initialize_episode)
    else
        Base.invokelatest(
            initialize_episode;
            split = case.split,
            base_seed = case.base_seed,
            mirror = case.mirror,
            offset = case.offset,
        )
    end
    return nothing
end

function normalize_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 && (values = dropdims(values; dims = 3))
    length(values) == 12 || error("Expected twelve actuator actions, got $(size(values)).")
    return vec(values)
end

function run_episode(protocol, case, action_function)
    initialize_case!(protocol, case)
    runtime_env = getfield(@__MODULE__, :env)
    nusselt_function = getfield(@__MODULE__, :state_Nu)
    rewards = Vector{Float64}(undef, TEST_STEPS)
    nusselt = Vector{Float64}(undef, TEST_STEPS)
    actions = Matrix{Float32}(undef, TEST_STEPS, 12)
    for step in 1:TEST_STEPS
        action = Base.invokelatest(action_function)
        actions[step, :] .= normalize_action(action)
        Base.invokelatest(runtime_env, action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
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
        case_spec = case,
        data_split = protocol === :fixed ? :shared : :test,
        steps = TEST_STEPS,
        rewards = episode.rewards,
        global_nusselt = episode.global_nusselt,
        actions = episode.actions,
        created_at = string(Dates.now()),
    )
    return episode
end

function load_test_manifest(options)
    isfile(options.manifest) || error("Frozen candidate manifest is missing: $(options.manifest)")
    manifest_hash = file_hash(options.manifest)
    manifest = JLD2.load(options.manifest)
    Bool(manifest["frozen_before_test"]) || error("Candidate manifest is not marked frozen.")
    Symbol(manifest["protocol"]) === options.protocol || error("Candidate manifest protocol mismatch.")
    candidates = [Dict{Symbol, Any}(Symbol(key) => value for (key, value) in raw) for raw in manifest["candidates"]]
    1 <= length(candidates) <= 2 || error("Expected one or two selected GO candidates.")
    expert_identifier = string(manifest["expert_identifier"])
    expert_path = local_expert_path(options.protocol, string(manifest["expert_path"]))
    return (; manifest_hash, candidates, expert_identifier, expert_path)
end

function configure_test_runtime!(options, expert_path, expert_identifier, output; runtime_tag = "sequential")
    ENV["DISTILLATION_PROTOCOL"] = string(options.protocol)
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    ENV["DISTILLATION_GROUP_CHANNELS"] = "false"
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV[options.protocol === :fixed ? "DISTILLATION_FIXED_EXPERT_PATH" : "DISTILLATION_VARYING_EXPERT_PATH"] = expert_path
    ENV["REVISION_RUN_SEED"] = string(P6_MASTER_SEED)
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output, "runtime", runtime_tag)
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output, "apprentice_output", runtime_tag)
    include(joinpath(DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
    expert_metadata = Base.invokelatest(
        () -> getfield(@__MODULE__, :DISTILLATION_EXPERT_METADATA),
    )
    loaded_identifier = string(expert_metadata[:identifier])
    loaded_identifier == expert_identifier || error("Local expert $loaded_identifier does not match study expert $expert_identifier.")
    return nothing
end

test_case_count(protocol) = protocol === :fixed ? 1 : 8

function parallel_episode_specs(protocol, candidates)
    return [
        (; controller_index, case_index)
        for controller_index in 0:length(candidates)
        for case_index in 1:test_case_count(protocol)
    ]
end

function controller_description(candidates, controller_index)
    controller_index == 0 && return (id = "expert", role = "expert", label = "MAT expert", candidate = nothing)
    1 <= controller_index <= length(candidates) || error("Invalid controller index $controller_index.")
    candidate = candidates[controller_index]
    role = string(candidate[:selection_role])
    label = @sprintf("%s: λ=%.4g, %d groups", role, Float64(candidate[:regularization_strength]), Int(candidate[:active_groups]))
    return (id = string(candidate[:candidate_id]), role, label, candidate)
end

episode_status_path(output, controller_index, case_index) = joinpath(
    output,
    "episode_status",
    @sprintf("c%02d_e%02d.jld2", controller_index, case_index),
)

function completed_episode_status(output, data, controller_index, case_index)
    status = load_status(episode_status_path(output, controller_index, case_index))
    isnothing(status) && return nothing
    get(status, :state, nothing) === :complete || return nothing
    controller = controller_description(data.candidates, controller_index)
    string(get(status, :controller_id, "")) == controller.id || return nothing
    haskey(status, :case) || return nothing
    case = status[:case]
    path = cache_path(output, controller.id, data.expert_identifier, case)
    isnothing(load_cache(path, controller.id, data.expert_identifier, case)) && return nothing
    return status
end

function run_single_test_episode(options, controller_index, case_index)
    data = load_test_manifest(options)
    output = joinpath(options.results_root, string(options.protocol), "analysis", "test")
    status_path = episode_status_path(output, controller_index, case_index)
    controller = controller_description(data.candidates, controller_index)
    write_status!(
        status_path;
        state = :running,
        protocol = options.protocol,
        controller_id = controller.id,
        controller_index,
        case_index,
        started_at = string(Dates.now()),
    )
    try
        runtime_tag = @sprintf("c%02d_e%02d", controller_index, case_index)
        configure_test_runtime!(options, data.expert_path, data.expert_identifier, output; runtime_tag)
        cases = test_cases(options.protocol)
        1 <= case_index <= length(cases) || error("Invalid case index $case_index for $(options.protocol).")
        case = cases[case_index]
        episodes = evaluate_controller!(
            output,
            options.results_root,
            options.protocol,
            [case],
            data.expert_identifier;
            candidate = controller.candidate,
        )
        path = cache_path(output, controller.id, data.expert_identifier, case)
        haskey(episodes, case_identifier(case)) || error("Episode cache was not produced for $(case_identifier(case)).")
        write_status!(
            status_path;
            state = :complete,
            protocol = options.protocol,
            controller_id = controller.id,
            controller_index,
            case_index,
            case,
            cache_path = path,
            completed_at = string(Dates.now()),
        )
        println("Terminal test episode complete: $(controller.id) / $(case_identifier(case))")
        return path
    catch error_value
        write_status!(
            status_path;
            state = :failed,
            protocol = options.protocol,
            controller_id = controller.id,
            controller_index,
            case_index,
            error_message = sprint(showerror, error_value),
            failed_at = string(Dates.now()),
        )
        rethrow()
    end
end

function run_parallel_test_episodes(options, data, output)
    mkpath(joinpath(output, "episode_logs"))
    launched = NamedTuple[]
    for spec in parallel_episode_specs(options.protocol, data.candidates)
        !isnothing(completed_episode_status(output, data, spec.controller_index, spec.case_index)) && continue
        log_path = joinpath(
            output,
            "episode_logs",
            @sprintf("c%02d_e%02d.log", spec.controller_index, spec.case_index),
        )
        log_io = open(log_path, "a")
        command = `$(Base.julia_cmd()) --startup-file=no --project=$PROJECT_ROOT $TEST_EPISODE_WORKER --protocol $(string(options.protocol)) --results-dir $(options.results_root) --manifest $(options.manifest) --controller-index $(spec.controller_index) --case-index $(spec.case_index)`
        try
            process = run(pipeline(command; stdout = log_io, stderr = log_io); wait = false)
            push!(launched, (; spec..., process, log_io, log_path))
        catch
            close(log_io)
            rethrow()
        end
    end
    println("Launched $(length(launched)) terminal test episode processes for $(options.protocol).")
    failed = NamedTuple[]
    for item in launched
        wait(item.process)
        close(item.log_io)
        success(item.process) || push!(failed, item)
    end
    isempty(failed) || error(
        "$(length(failed)) parallel terminal test episode worker(s) failed: " *
        join((item.log_path for item in failed), ", "),
    )
    return nothing
end

function load_parallel_test_outputs(options, data, output)
    specs = parallel_episode_specs(options.protocol, data.candidates)
    statuses = Dict{Tuple{Int, Int}, Any}()
    for spec in specs
        status = completed_episode_status(output, data, spec.controller_index, spec.case_index)
        isnothing(status) && error("Missing or invalid completed episode c$(spec.controller_index)/e$(spec.case_index).")
        statuses[(spec.controller_index, spec.case_index)] = status
    end
    cases = [statuses[(0, case_index)][:case] for case_index in 1:test_case_count(options.protocol)]
    controllers = NamedTuple[]
    for controller_index in 0:length(data.candidates)
        controller = controller_description(data.candidates, controller_index)
        episodes = Dict{String, Any}()
        for case in cases
            path = cache_path(output, controller.id, data.expert_identifier, case)
            episode = load_cache(path, controller.id, data.expert_identifier, case)
            isnothing(episode) && error("Missing or invalid cache for $(controller.id) / $(case_identifier(case)).")
            episodes[case_identifier(case)] = episode
        end
        push!(controllers, (; controller..., episodes))
    end
    return cases, controllers
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
        runtime_flux = getfield(@__MODULE__, :Flux)
        Base.invokelatest(runtime_flux.testmode!, candidate_model)
        input_mask = Float32.(candidate[:mask])
    end
    episodes = Dict{String, Any}()
    for case in cases
        path = cache_path(output, controller_id, expert_identifier, case)
        episode = load_cache(path, controller_id, expert_identifier, case)
        if isnothing(episode)
            println("  $controller_id: $(case_identifier(case))")
            action_function = if isnothing(candidate)
                runtime_rl = getfield(@__MODULE__, :RL)
                runtime_agent = getfield(@__MODULE__, :agent)
                runtime_env = getfield(@__MODULE__, :env)
                () -> Base.invokelatest(runtime_rl.prob, runtime_agent.policy, runtime_env).μ
            else
                runtime_rl = getfield(@__MODULE__, :RL)
                runtime_env = getfield(@__MODULE__, :env)
                () -> Base.invokelatest(
                    runtime_rl.prob,
                    candidate_model,
                    runtime_env.state .* input_mask,
                    nothing,
                ).μ[:, :, 1]
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

function finalize_test_results(options, data, output, cases, controllers)
    file_hash(options.manifest) == data.manifest_hash || error("Candidate manifest changed during test execution.")
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
        parallel_episodes = options.parallel_test,
        candidate_manifest = options.manifest,
        candidate_manifest_sha256 = data.manifest_hash,
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

function run_test_worker(options)
    data = load_test_manifest(options)
    output = joinpath(options.results_root, string(options.protocol), "analysis", "test")
    cases, controllers = if options.parallel_test
        run_parallel_test_episodes(options, data, output)
        load_parallel_test_outputs(options, data, output)
    else
        configure_test_runtime!(options, data.expert_path, data.expert_identifier, output)
        sequential_cases = test_cases(options.protocol)
        sequential_controllers = NamedTuple[]
        for controller_index in 0:length(data.candidates)
            controller = controller_description(data.candidates, controller_index)
            episodes = evaluate_controller!(
                output,
                options.results_root,
                options.protocol,
                sequential_cases,
                data.expert_identifier;
                candidate = controller.candidate,
            )
            push!(sequential_controllers, (; controller..., episodes))
        end
        sequential_cases, sequential_controllers
    end
    return finalize_test_results(options, data, output, cases, controllers)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        run_test_worker(parse_arguments(ARGS))
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
