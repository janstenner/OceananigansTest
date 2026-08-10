using Dates
using JLD2
using PlotlyJS
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "inspect_strength_calibration_pilot.jl"))

const P6_CALIBRATION_TEST_SCHEMA_VERSION = 1
const P6_CALIBRATION_TEST_EPISODE_STEPS = 200
const P6_CALIBRATION_TEST_DIAGNOSTIC_ROOT = joinpath(
    P6_CALIBRATION_ANALYSIS_DIRECTORY,
    "calibration_test_diagnostic",
)
const P6_CALIBRATION_TEST_LINE_DASHES = (
    "solid",
    "dash",
    "dot",
    "dashdot",
    "longdash",
    "longdashdot",
)

function calibration_test_usage(io::IO = stdout)
    println(io, """
    Internal closed-loop calibration test worker.

    Usage:
      julia --project=. inspect_strength_calibration_test_worker.jl \\
        --protocol fixed|varying \\
        --grouping grouped_channels|separate_channels

    Run inspect_strength_calibration_pilot.jl instead of this helper directly.
    """)
end

function parse_calibration_test_arguments(arguments)
    options = Dict{String, Any}(
        "protocol" => nothing,
        "grouping" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            calibration_test_usage()
            return nothing
        elseif startswith(argument, "--")
            index == length(arguments) && error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(options, key) || error("Unknown option '$argument'.")
            options[key] = arguments[index + 1]
            index += 1
        else
            error("Unknown argument '$argument'.")
        end
        index += 1
    end

    isnothing(options["protocol"]) && error("--protocol is required.")
    isnothing(options["grouping"]) && error("--grouping is required.")
    protocol = Symbol(lowercase(string(options["protocol"])))
    grouping = Symbol(lowercase(string(options["grouping"])))
    protocol in (:fixed, :varying) || error("--protocol must be fixed or varying.")
    grouping in (:grouped_channels, :separate_channels) || error(
        "--grouping must be grouped_channels or separate_channels.",
    )
    return (; protocol, grouping)
end

function calibration_test_atomic_save(path::AbstractString; entries...)
    mkpath(dirname(path))
    temporary_path = path * ".tmp.$(getpid()).$(time_ns())"
    try
        JLD2.jldopen(temporary_path, "w") do file
            for (key, value) in pairs(entries)
                file[string(key)] = value
            end
        end
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return path
end

function calibration_test_cache_tag(material::AbstractString)
    return bytes2hex(SHA.sha256(codeunits(material)))[1:20]
end

function calibration_test_case_identifier(case)
    case === nothing && return "fixed_shared"
    return "$(case.split)_base_$(case.base_seed)_mirror_$(case.mirror ? 1 : 0)_offset_$(case.offset)"
end

function calibration_test_cases(protocol::Symbol)
    protocol === :fixed && return [nothing]
    corpus = getfield(@__MODULE__, :CORPUS)
    split = corpus[:test]
    base_seeds = sort!(Int.(collect(keys(split))))
    length(base_seeds) == 2 || error(
        "Expected two Varying-IC test bases, found $(length(base_seeds)).",
    )
    return vec([
        (split = :test, base_seed, mirror, offset)
        for base_seed in base_seeds,
            mirror in (false, true),
            offset in (0, 20)
    ])
end

function normalize_calibration_test_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 &&
        (values = dropdims(values; dims = 3))
    length(values) == 12 || error(
        "Expected twelve actuator actions, got size $(size(values)).",
    )
    return vec(values)
end

function initialize_calibration_test_case!(protocol::Symbol, case)
    runtime_rl = getfield(@__MODULE__, :RL)
    runtime_env = getfield(@__MODULE__, :env)
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    if protocol === :fixed
        Base.invokelatest(getfield(@__MODULE__, :generate_random_init))
    else
        Base.invokelatest(
            getfield(@__MODULE__, :generate_random_init);
            split = case.split,
            base_seed = case.base_seed,
            mirror = case.mirror,
            offset = case.offset,
        )
    end
    return nothing
end

function run_calibration_test_episode(
    protocol::Symbol,
    case,
    action_function::Function;
    episode_steps::Int = P6_CALIBRATION_TEST_EPISODE_STEPS,
)
    initialize_calibration_test_case!(protocol, case)
    runtime_env = getfield(@__MODULE__, :env)
    nusselt_function = getfield(@__MODULE__, :state_Nu)
    rewards = Vector{Float64}(undef, episode_steps)
    global_nusselt = Vector{Float64}(undef, episode_steps)
    actions = Matrix{Float32}(undef, episode_steps, 12)
    for step in 1:episode_steps
        action = Base.invokelatest(action_function)
        actions[step, :] .= normalize_calibration_test_action(action)
        Base.invokelatest(runtime_env, action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        global_nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
        isfinite(rewards[step]) || error("Non-finite reward at test step $step.")
        isfinite(global_nusselt[step]) || error("Non-finite Nusselt number at test step $step.")
    end
    return (; rewards, global_nusselt, actions)
end

function load_calibration_test_cache(
    path::AbstractString;
    controller::Symbol,
    expert_identifier::AbstractString,
    case_identifier::AbstractString,
    candidate_id = nothing,
)
    isfile(path) || return nothing
    try
        loaded = JLD2.load(path)
        Int(loaded["schema_version"]) == P6_CALIBRATION_TEST_SCHEMA_VERSION || return nothing
        Symbol(loaded["diagnostic_scope"]) === :calibration_test_diagnostic || return nothing
        Symbol(loaded["controller"]) === controller || return nothing
        string(loaded["expert_identifier"]) == expert_identifier || return nothing
        string(loaded["case_identifier"]) == case_identifier || return nothing
        Int(loaded["episode_steps"]) == P6_CALIBRATION_TEST_EPISODE_STEPS || return nothing
        if !isnothing(candidate_id)
            haskey(loaded, "candidate_id") || return nothing
            string(loaded["candidate_id"]) == string(candidate_id) || return nothing
        end
        return Dict{Symbol, Any}(
            :rewards => Float64.(loaded["rewards"]),
            :global_nusselt => Float64.(loaded["global_nusselt"]),
            :actions => Float32.(loaded["actions"]),
        )
    catch
        return nothing
    end
end

function save_calibration_test_cache!(
    path::AbstractString,
    episode;
    controller::Symbol,
    protocol::Symbol,
    grouping::Symbol,
    expert_identifier::AbstractString,
    case,
    candidate = nothing,
)
    case_identifier = calibration_test_case_identifier(case)
    entries = Dict{Symbol, Any}(
        :schema_version => P6_CALIBRATION_TEST_SCHEMA_VERSION,
        :diagnostic_scope => :calibration_test_diagnostic,
        :scientific_selection_allowed => false,
        :data_split => protocol === :fixed ? :shared : :test,
        :controller => controller,
        :protocol => protocol,
        :grouping => grouping,
        :expert_identifier => String(expert_identifier),
        :case_identifier => case_identifier,
        :case => case,
        :episode_steps => P6_CALIBRATION_TEST_EPISODE_STEPS,
        :rewards => episode.rewards,
        :global_nusselt => episode.global_nusselt,
        :actions => episode.actions,
        :created_at => string(Dates.now()),
    )
    if !isnothing(candidate)
        entries[:candidate_id] = string(candidate[:candidate_id])
        entries[:candidate_update] = Int(candidate[:update])
        entries[:regularized_updates] = Int(candidate[:regularized_updates])
        entries[:active_groups] = Int(candidate[:active_groups])
        entries[:regularization_strength] = Float64(candidate[:calibration_strength])
        entries[:regression_learning_rate] = Float64(candidate[:regression_learning_rate])
    end
    calibration_test_atomic_save(path; entries...)
    return Dict{Symbol, Any}(
        :rewards => Float64.(episode.rewards),
        :global_nusselt => Float64.(episode.global_nusselt),
        :actions => Float32.(episode.actions),
    )
end

function resolve_calibration_candidate_checkpoint(candidate)
    recorded_path = get(candidate, :model_path, nothing)
    if !isnothing(recorded_path) && isfile(string(recorded_path))
        return abspath(string(recorded_path))
    end
    isnothing(recorded_path) && error(
        "Pareto candidate $(candidate[:candidate_id]) has no model path.",
    )
    run_directory = string(candidate[:source_run_directory])
    filename = basename(replace(string(recorded_path), '\\' => '/'))
    local_path = joinpath(run_directory, "candidates", filename)
    isfile(local_path) || error(
        "Pareto candidate checkpoint is missing at '$recorded_path' and '$local_path'.",
    )
    return abspath(local_path)
end

function resolve_calibration_expert_checkpoint(
    protocol::Symbol,
    recorded_path::AbstractString,
)
    isfile(recorded_path) && return abspath(recorded_path)

    local_path = normpath(joinpath(
        @__DIR__,
        "..",
        "Expert_Apprentice_Distillation",
        "experts",
        string(protocol),
        "agent.jld2",
    ))
    isfile(local_path) || error(
        "Expert checkpoint is missing at both the recorded and local paths: " *
        "'$recorded_path' and '$local_path'.",
    )
    println("  Recorded server expert is unavailable; using local checkpoint: $local_path")
    return abspath(local_path)
end

function configure_calibration_test_runtime!(options, loaded)
    expert_identifiers = unique(string(run.config[:expert_identifier]) for run in loaded.runs)
    expert_paths = unique(string(run.config[:expert_path]) for run in loaded.runs)
    length(expert_identifiers) == 1 || error("Calibration runs use different experts.")
    length(expert_paths) == 1 || error("Calibration runs use different expert paths.")
    expected_expert_identifier = only(expert_identifiers)
    expert_path = resolve_calibration_expert_checkpoint(
        options.protocol,
        only(expert_paths),
    )
    group_channels = options.grouping === :grouped_channels
    output_directory = joinpath(
        P6_CALIBRATION_TEST_DIAGNOSTIC_ROOT,
        string(options.protocol),
        string(options.grouping),
    )
    ENV["DISTILLATION_PROTOCOL"] = string(options.protocol)
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(group_channels)
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV[options.protocol === :fixed ? "DISTILLATION_FIXED_EXPERT_PATH" : "DISTILLATION_VARYING_EXPERT_PATH"] = expert_path
    ENV["REVISION_RUN_SEED"] = string(only(unique(Int(run.config[:apprentice_seed]) for run in loaded.runs)))
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output_directory, "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output_directory, "apprentice_outputs")
    include(joinpath(
        @__DIR__,
        "..",
        "Expert_Apprentice_Distillation",
        "Expert_Apprentice.jl",
    ))
    expert_metadata = Base.invokelatest(
        () -> getfield(@__MODULE__, :DISTILLATION_EXPERT_METADATA),
    )
    loaded_expert_identifier = string(expert_metadata[:identifier])
    loaded_expert_identifier == expected_expert_identifier || error(
        "Local expert '$loaded_expert_identifier' does not match the calibration " *
        "expert '$expected_expert_identifier'.",
    )
    return (
        expert_identifier = expected_expert_identifier,
        output_directory,
    )
end

function controller_cache_path(
    protocol::Symbol,
    grouping::Symbol,
    controller::Symbol,
    expert_identifier::AbstractString,
    case,
    candidate = nothing,
)
    case_identifier = calibration_test_case_identifier(case)
    identity = isnothing(candidate) ? "expert" : string(candidate[:candidate_id])
    grouping_identity = isnothing(candidate) ? "shared_expert" : string(grouping)
    material = join((
        string(P6_CALIBRATION_TEST_SCHEMA_VERSION),
        string(protocol),
        grouping_identity,
        string(controller),
        expert_identifier,
        case_identifier,
        identity,
        string(P6_CALIBRATION_TEST_EPISODE_STEPS),
    ), '|')
    cache_directory = joinpath(
        P6_CALIBRATION_TEST_DIAGNOSTIC_ROOT,
        "cache",
        string(protocol),
        isnothing(candidate) ? "expert" : string(grouping),
    )
    return joinpath(cache_directory, "$(calibration_test_cache_tag(material)).jld2")
end

function retained_calibration_pareto_candidates(runs)
    candidates = Dict{Symbol, Any}[]
    seen_ids = Set{String}()
    for run in runs, raw_candidate in run.pareto_front
        candidate = copy(raw_candidate)
        candidate_id = string(candidate[:candidate_id])
        candidate_id in seen_ids && continue
        push!(seen_ids, candidate_id)
        push!(candidates, candidate)
    end
    sort!(candidates; by = candidate -> (
        Float64(candidate[:calibration_strength]),
        -Int(candidate[:active_groups]),
        Int(candidate[:update]),
    ))
    return candidates
end

function evaluate_expert_test_cases!(options, cases, expert_identifier)
    episodes = Dict{String, Dict{Symbol, Any}}()
    for case in cases
        case_identifier = calibration_test_case_identifier(case)
        path = controller_cache_path(
            options.protocol,
            options.grouping,
            :expert,
            expert_identifier,
            case,
        )
        episode = load_calibration_test_cache(
            path;
            controller = :expert,
            expert_identifier,
            case_identifier,
        )
        if isnothing(episode)
            println("  expert: $case_identifier")
            action_function = () -> begin
                runtime_rl = getfield(@__MODULE__, :RL)
                runtime_agent = getfield(@__MODULE__, :agent)
                runtime_env = getfield(@__MODULE__, :env)
                Base.invokelatest(runtime_rl.prob, runtime_agent.policy, runtime_env).μ
            end
            generated = Base.invokelatest(
                run_calibration_test_episode,
                options.protocol,
                case,
                action_function,
            )
            episode = save_calibration_test_cache!(
                path,
                generated;
                controller = :expert,
                protocol = options.protocol,
                grouping = options.grouping,
                expert_identifier,
                case,
            )
        end
        episodes[case_identifier] = episode
    end
    return episodes
end

function evaluate_candidate_test_cases!(options, candidate, cases, expert_identifier)
    checkpoint_path = resolve_calibration_candidate_checkpoint(candidate)
    loaded_checkpoint = JLD2.load(checkpoint_path)
    haskey(loaded_checkpoint, "model_payload") || error(
        "Candidate checkpoint has no model_payload: $checkpoint_path",
    )
    candidate_model = loaded_checkpoint["model_payload"]
    runtime_flux = getfield(@__MODULE__, :Flux)
    Base.invokelatest(runtime_flux.testmode!, candidate_model)
    input_mask = Float32.(candidate[:mask])
    episodes = Dict{String, Dict{Symbol, Any}}()
    for case in cases
        case_identifier = calibration_test_case_identifier(case)
        path = controller_cache_path(
            options.protocol,
            options.grouping,
            :apprentice,
            expert_identifier,
            case,
            candidate,
        )
        episode = load_calibration_test_cache(
            path;
            controller = :apprentice,
            expert_identifier,
            case_identifier,
            candidate_id = candidate[:candidate_id],
        )
        if isnothing(episode)
            println(
                "  candidate $(candidate[:candidate_id]), groups=$(candidate[:active_groups]): " *
                case_identifier,
            )
            action_function = () -> begin
                runtime_rl = getfield(@__MODULE__, :RL)
                runtime_env = getfield(@__MODULE__, :env)
                Base.invokelatest(
                    runtime_rl.prob,
                    candidate_model,
                    runtime_env.state .* input_mask,
                    nothing,
                ).μ[:, :, 1]
            end
            generated = Base.invokelatest(
                run_calibration_test_episode,
                options.protocol,
                case,
                action_function,
            )
            episode = save_calibration_test_cache!(
                path,
                generated;
                controller = :apprentice,
                protocol = options.protocol,
                grouping = options.grouping,
                expert_identifier,
                case,
                candidate,
            )
        end
        episodes[case_identifier] = episode
    end
    return episodes
end

function episode_matrix(episodes, cases, field::Symbol)
    return reduce(
        vcat,
        [permutedims(Float64.(episodes[calibration_test_case_identifier(case)][field])) for case in cases],
    )
end

function candidate_curve_label(candidate)
    return @sprintf(
        "λ=%.6g, groups=%d, update=%d/%d",
        Float64(candidate[:calibration_strength]),
        Int(candidate[:active_groups]),
        Int(candidate[:update]),
        Int(candidate[:regularized_updates]),
    )
end

function plot_calibration_test_reward_curves(
    options,
    loaded,
    cases,
    expert_episodes,
    candidate_results,
    output_directory,
)
    expert_matrix = episode_matrix(expert_episodes, cases, :rewards)
    steps = collect(1:size(expert_matrix, 2))
    traces = PlotlyJS.GenericTrace[]
    if size(expert_matrix, 1) > 1
        for case_index in axes(expert_matrix, 1)
            push!(traces, scatter(
                x = steps,
                y = vec(expert_matrix[case_index, :]),
                mode = "lines",
                name = "Expert case $case_index",
                showlegend = false,
                line = attr(color = "rgba(70,70,70,0.22)", width = 1),
                hovertemplate = "Expert case $case_index<br>Step %{x}<br>Reward %{y:.6f}<extra></extra>",
            ))
        end
    end
    push!(traces, scatter(
        x = steps,
        y = vec(mean(expert_matrix; dims = 1)),
        mode = "lines",
        name = size(expert_matrix, 1) == 1 ? "MAT expert" : "MAT expert mean",
        line = attr(color = "#202020", width = 3.5),
        hovertemplate = "Expert mean<br>Step %{x}<br>Reward %{y:.6f}<extra></extra>",
    ))

    per_strength_index = Dict{Float64, Int}()
    for result in candidate_results
        candidate = result.candidate
        strength = Float64(candidate[:calibration_strength])
        dash_index = get(per_strength_index, strength, 0) + 1
        per_strength_index[strength] = dash_index
        matrix = episode_matrix(result.episodes, cases, :rewards)
        minimum_rewards = vec(minimum(matrix; dims = 1))
        maximum_rewards = vec(maximum(matrix; dims = 1))
        push!(traces, scatter(
            x = steps,
            y = vec(mean(matrix; dims = 1)),
            mode = "lines",
            name = candidate_curve_label(candidate),
            customdata = hcat(minimum_rewards, maximum_rewards),
            line = attr(
                color = calibration_strength_color(strength, loaded.runs),
                width = 2.5,
                dash = P6_CALIBRATION_TEST_LINE_DASHES[mod1(dash_index, length(P6_CALIBRATION_TEST_LINE_DASHES))],
            ),
            hovertemplate =
                "Step %{x}<br>Mean reward %{y:.6f}<br>" *
                "Case range [%{customdata[0]:.6f}, %{customdata[1]:.6f}]<extra>%{fullData.name}</extra>",
        ))
    end

    title = "$(calibration_protocol_label(options.protocol)) — " *
            "$(calibration_grouping_label(options.grouping)) — test reward curves"
    plot_handle = Plot(traces, Layout(
        template = "plotly_white",
        title = attr(text = title, x = 0.5, xanchor = "center"),
        width = 1050,
        height = 620,
        margin = attr(l = 95, r = 35, t = 80, b = 75),
        font = attr(family = "Arial, sans-serif", size = 14, color = "#303030"),
        xaxis = attr(title = "Control step", gridcolor = "#E6E6E6", zeroline = false),
        yaxis = attr(
            title = "Mean environment reward (higher is better)",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        legend = attr(
            x = 0.01,
            y = 0.01,
            xanchor = "left",
            yanchor = "bottom",
            bgcolor = "rgba(255,255,255,0.92)",
            bordercolor = "#CFCFCF",
            borderwidth = 1,
            font = attr(size = 11),
        ),
        hovermode = "x unified",
    ))
    output_path = joinpath(
        output_directory,
        "$(options.protocol)_$(options.grouping)_test_reward_curves.svg",
    )
    PlotlyJS.savefig(plot_handle, output_path; width = 1050, height = 620)
    return output_path
end

function plot_calibration_test_return_boxes(
    options,
    loaded,
    cases,
    expert_episodes,
    candidate_results,
    output_directory,
)
    traces = PlotlyJS.GenericTrace[]
    expert_returns = [
        sum(expert_episodes[calibration_test_case_identifier(case)][:rewards])
        for case in cases
    ]
    push!(traces, box(
        y = expert_returns,
        name = "MAT expert",
        boxpoints = "all",
        quartilemethod = "linear",
        boxmean = true,
        marker = attr(color = "#202020", size = 7),
        line = attr(color = "#202020", width = 2),
        hovertemplate = "Return %{y:.6f}<extra>%{fullData.name}</extra>",
    ))
    for result in candidate_results
        candidate = result.candidate
        returns = [
            sum(result.episodes[calibration_test_case_identifier(case)][:rewards])
            for case in cases
        ]
        color = calibration_strength_color(candidate[:calibration_strength], loaded.runs)
        push!(traces, box(
            y = returns,
            name = candidate_curve_label(candidate),
            boxpoints = "all",
            quartilemethod = "linear",
            boxmean = true,
            marker = attr(color = color, size = 7),
            line = attr(color = color, width = 2),
            hovertemplate = "Return %{y:.6f}<extra>%{fullData.name}</extra>",
        ))
    end
    title = "$(calibration_protocol_label(options.protocol)) — " *
            "$(calibration_grouping_label(options.grouping)) — test episode returns"
    plot_handle = Plot(traces, Layout(
        template = "plotly_white",
        title = attr(text = title, x = 0.5, xanchor = "center"),
        width = 1150,
        height = 650,
        margin = attr(l = 100, r = 35, t = 80, b = 180),
        font = attr(family = "Arial, sans-serif", size = 13, color = "#303030"),
        xaxis = attr(tickangle = -25),
        yaxis = attr(
            title = "Cumulative 200-step reward (higher is better)",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        showlegend = false,
    ))
    output_path = joinpath(
        output_directory,
        "$(options.protocol)_$(options.grouping)_test_return_boxplot.svg",
    )
    PlotlyJS.savefig(plot_handle, output_path; width = 1150, height = 650)
    return output_path
end

function write_calibration_test_csv(
    path,
    options,
    cases,
    expert_episodes,
    candidate_results,
)
    open(path, "w") do io
        println(io, "protocol,grouping,controller,strength,active_groups,update,regularized_updates,regression_learning_rate,candidate_id,test_case,total_reward,mean_reward")
        for case in cases
            identifier = calibration_test_case_identifier(case)
            rewards = expert_episodes[identifier][:rewards]
            println(io, join((
                options.protocol,
                options.grouping,
                "expert",
                "",
                "",
                "",
                "",
                "",
                "",
                identifier,
                sum(rewards),
                mean(rewards),
            ), ','))
        end
        for result in candidate_results, case in cases
            candidate = result.candidate
            identifier = calibration_test_case_identifier(case)
            rewards = result.episodes[identifier][:rewards]
            println(io, join((
                options.protocol,
                options.grouping,
                "apprentice",
                candidate[:calibration_strength],
                candidate[:active_groups],
                candidate[:update],
                candidate[:regularized_updates],
                candidate[:regression_learning_rate],
                candidate[:candidate_id],
                identifier,
                sum(rewards),
                mean(rewards),
            ), ','))
        end
    end
    return path
end

function run_loaded_calibration_test_worker(options, loaded, runtime)
    loaded.study_complete || error(
        "The long-budget calibration block is incomplete for " *
        "$(options.protocol)/$(options.grouping). Refusing to run a partial test diagnostic.",
    )
    candidates = retained_calibration_pareto_candidates(loaded.runs)
    isempty(candidates) && error("The retained calibration Pareto archives are empty.")
    for candidate in candidates
        resolve_calibration_candidate_checkpoint(candidate)
    end
    cases = calibration_test_cases(options.protocol)
    mkpath(runtime.output_directory)
    println(
        "Closed-loop test diagnostic: $(options.protocol)/$(options.grouping), " *
        "$(length(candidates)) retained Pareto candidates × $(length(cases)) cases",
    )
    expert_episodes = evaluate_expert_test_cases!(
        options,
        cases,
        runtime.expert_identifier,
    )
    candidate_results = NamedTuple[]
    for candidate in candidates
        episodes = evaluate_candidate_test_cases!(
            options,
            candidate,
            cases,
            runtime.expert_identifier,
        )
        push!(candidate_results, (; candidate, episodes))
    end

    aggregate_path = joinpath(runtime.output_directory, "closed_loop_test_results.jld2")
    calibration_test_atomic_save(
        aggregate_path;
        schema_version = P6_CALIBRATION_TEST_SCHEMA_VERSION,
        diagnostic_scope = :calibration_test_diagnostic,
        scientific_selection_allowed = false,
        test_split_consumed = options.protocol === :varying,
        protocol = options.protocol,
        grouping = options.grouping,
        regularized_update_budgets = loaded.budgets,
        expert_identifier = runtime.expert_identifier,
        cases,
        candidates,
        expert_episodes,
        candidate_results,
        completed_at = string(Dates.now()),
    )
    curve_path = plot_calibration_test_reward_curves(
        options,
        loaded,
        cases,
        expert_episodes,
        candidate_results,
        runtime.output_directory,
    )
    box_path = plot_calibration_test_return_boxes(
        options,
        loaded,
        cases,
        expert_episodes,
        candidate_results,
        runtime.output_directory,
    )
    csv_path = write_calibration_test_csv(
        joinpath(runtime.output_directory, "closed_loop_test_returns.csv"),
        options,
        cases,
        expert_episodes,
        candidate_results,
    )
    println("  $curve_path")
    println("  $box_path")
    println("  $csv_path")
    println("  $aggregate_path")
    return nothing
end

function calibration_test_worker_main(arguments = ARGS)
    options = parse_calibration_test_arguments(arguments)
    isnothing(options) && return nothing
    loaded = load_calibration_combination(options.protocol, options.grouping)
    loaded.study_complete || error(
        "Long-budget calibration block is not complete for " *
        "$(options.protocol)/$(options.grouping).",
    )
    runtime = configure_calibration_test_runtime!(options, loaded)
    return Base.invokelatest(run_loaded_calibration_test_worker, options, loaded, runtime)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        calibration_test_worker_main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
