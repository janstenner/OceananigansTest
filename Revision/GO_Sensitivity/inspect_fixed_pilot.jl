using JLD2
using PlotlyJS
using Printf
using Statistics

# Tabular/Pareto inspection loads only lightweight result files. The optional
# closed-loop comparison initializes the Fixed-IC runtime and deserializes the
# expert and selected apprentice only when their episode caches are absent.
if !isdefined(@__MODULE__, :PARETO_ARCHIVE_SCHEMA_VERSION)
    include(joinpath(
        @__DIR__,
        "..",
        "Expert_Apprentice_Distillation",
        "ParetoArchive.jl",
    ))
end

const P6_FIXED_PILOT_INSPECTION_ROOT = joinpath(@__DIR__, "results", "fixed_pilot")
const P6_CLOSED_LOOP_SCHEMA_VERSION = 1
const P6_CLOSED_LOOP_EPISODE_STEPS = 200
const P6_PLOT_COLORS = Dict(
    :native => "#2166AC",
    :pilot_relative_001 => "#D6604D",
    :pilot_keep_6_groups => "#4D9221",
    :pilot_keep_16_groups => "#4D9221",
)
const P6_PLOT_SYMBOLS = Dict(
    :native => "circle",
    :pilot_relative_001 => "diamond",
    :pilot_keep_6_groups => "square",
    :pilot_keep_16_groups => "square",
)

function latest_fixed_pilot_run_directory()
    isdir(P6_FIXED_PILOT_INSPECTION_ROOT) || error(
        "No Fixed-IC pilot result directory exists at $P6_FIXED_PILOT_INSPECTION_ROOT.",
    )
    directories = filter(
        isdir,
        readdir(P6_FIXED_PILOT_INSPECTION_ROOT; join = true),
    )
    isempty(directories) && error(
        "No Fixed-IC pilot runs exist below $P6_FIXED_PILOT_INSPECTION_ROOT.",
    )
    completed = filter(
        directory -> isfile(joinpath(directory, "pilot_summary.jld2")),
        directories,
    )
    candidates = isempty(completed) ? directories : completed
    return last(sort!(candidates; by = mtime))
end

function fixed_pilot_evaluation_files(run_directory::AbstractString)
    directory = joinpath(run_directory, "evaluations")
    isdir(directory) || error("The run has no evaluation directory: $directory")
    files = sort!(filter(
        path -> startswith(basename(path), "update_") && endswith(path, ".jld2"),
        readdir(directory; join = true),
    ))
    isempty(files) && error("The run has no evaluation shards: $directory")
    return files
end

function csv_cell(value)
    value === nothing && return ""
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function write_candidate_csv(path::AbstractString, records)
    columns = (
        :update,
        :threshold_id,
        :threshold_kind,
        :pareto_scope,
        :analysis_scope,
        :active_inputs,
        :active_groups,
        :active_sensor_locations,
        :validation_matching,
        :teacher_forced_validation_matching,
        :numeric_status,
        :candidate_id,
    )
    open(path, "w") do io
        println(io, join((
            string.(columns)...,
            "recorded_loadable",
            "model_file_exists_now",
            "recorded_model_path",
        ), ','))
        for record in records
            values = [archive_value(record, column; default = nothing) for column in columns]
            model_path = archive_value(record, :model_path; default = nothing)
            recorded_loadable = Bool(archive_value(record, :loadable; default = false))
            model_file_exists = !isnothing(model_path) && isfile(string(model_path))
            append!(values, (recorded_loadable, model_file_exists, model_path))
            println(io, join(csv_cell.(values), ','))
        end
    end
    return path
end

function pilot_record_sort_key(record)
    threshold_id = Symbol(archive_value(record, :threshold_id))
    threshold_rank = threshold_id === :native ? 0 : 1
    return (
        Int(archive_value(record, :update)),
        threshold_rank,
        string(threshold_id),
    )
end

function print_candidate_row(io::IO, record)
    @printf(
        io,
        "  %6d  %-24s  %6d  %6d  %7d  %14.8e  %14.8e\n",
        Int(archive_value(record, :update)),
        string(archive_value(record, :threshold_id)),
        Int(archive_value(record, :active_groups)),
        Int(archive_value(record, :active_inputs)),
        Int(archive_value(record, :active_sensor_locations)),
        Float64(archive_value(record, :validation_matching)),
        Float64(archive_value(record, :teacher_forced_validation_matching)),
    )
end

function print_front(io::IO, heading::AbstractString, records)
    println(io, heading)
    if isempty(records)
        println(io, "  (empty)")
        return
    end
    println(io, "  update  threshold                 groups  inputs  sensors   autoreg. MSE    teacher MSE")
    println(io, "  ------  ------------------------  ------  ------  -------  --------------  --------------")
    for record in sort!(collect(records); by = pilot_record_sort_key)
        print_candidate_row(io, record)
    end
end

function fixed_pilot_threshold_ids(records)
    ids = unique(Symbol(archive_value(record, :threshold_id)) for record in records)
    return sort!(collect(ids); by = id -> (id === :native ? 0 : 1, string(id)))
end

function plot_fixed_pilot_validation(records, current_front, analysis_directory)
    traces = PlotlyJS.GenericTrace[]
    for threshold_id in fixed_pilot_threshold_ids(records)
        threshold_records = sort!(
            filter(
                record -> Symbol(archive_value(record, :threshold_id)) === threshold_id,
                records,
            );
            by = record -> Int(archive_value(record, :update)),
        )
        updates = Int[archive_value(record, :update) for record in threshold_records]
        active_inputs = Int[archive_value(record, :active_inputs) for record in threshold_records]
        active_groups = Int[archive_value(record, :active_groups) for record in threshold_records]
        active_sensors = Int[
            archive_value(record, :active_sensor_locations) for record in threshold_records
        ]
        validation_loss = Float64[
            archive_value(record, :validation_matching) for record in threshold_records
        ]
        color = get(P6_PLOT_COLORS, threshold_id, "#777777")
        push!(
            traces,
            scatter(
                x = active_groups,
                y = validation_loss,
                mode = "markers",
                name = string(threshold_id),
                customdata = hcat(updates, active_inputs, active_sensors),
                marker = attr(
                    color = color,
                    size = threshold_id === :native ? 8 : 7,
                    symbol = get(P6_PLOT_SYMBOLS, threshold_id, "circle"),
                    line = attr(color = "white", width = 0.8),
                ),
                hovertemplate =
                    "Threshold: %{fullData.name}<br>" *
                    "Update: %{customdata[0]}<br>" *
                    "Active groups: %{x}<br>" *
                    "Active global inputs: %{customdata[1]}<br>" *
                    "Active global sensor locations: %{customdata[2]}<br>" *
                    "Validation MSE: %{y:.6e}<extra></extra>",
            ),
        )
    end

    if !isempty(current_front)
        push!(
            traces,
            scatter(
                x = Int[
                    archive_value(record, :active_groups) for record in current_front
                ],
                y = Float64[
                    archive_value(record, :validation_matching) for record in current_front
                ],
                mode = "markers",
                name = "retained archive",
                customdata = hcat(
                    Int[archive_value(record, :update) for record in current_front],
                    string.(archive_value.(current_front, Ref(:threshold_id))),
                    Int[
                        archive_value(record, :active_inputs) for record in current_front
                    ],
                    Int[
                        archive_value(record, :active_sensor_locations) for record in current_front
                    ],
                ),
                marker = attr(
                    color = "rgba(0, 0, 0, 0)",
                    size = 15,
                    symbol = "circle",
                    line = attr(color = "#111111", width = 2.5),
                ),
                hovertemplate =
                    "Retained: %{customdata[1]}<br>" *
                    "Update: %{customdata[0]}<br>" *
                    "Active groups: %{x}<br>" *
                    "Active global inputs: %{customdata[2]}<br>" *
                    "Active global sensor locations: %{customdata[3]}<br>" *
                    "Validation MSE: %{y:.6e}<extra></extra>",
            ),
        )
    end

    plot_handle = Plot(
        traces,
        Layout(
            template = "plotly_white",
            title = attr(
                text = "Fixed-IC GO pilot — validation checkpoints",
                x = 0.5,
                xanchor = "center",
                font = attr(size = 22, color = "#252525"),
            ),
            paper_bgcolor = "white",
            plot_bgcolor = "white",
            width = 900,
            height = 560,
            margin = attr(l = 95, r = 30, t = 80, b = 80),
            font = attr(family = "Arial, sans-serif", size = 15, color = "#303030"),
            xaxis = attr(
                title = attr(text = "Active groups (lower is sparser)", standoff = 12),
                showline = true,
                mirror = true,
                linecolor = "#3A3A3A",
                linewidth = 1,
                ticks = "outside",
                gridcolor = "#E6E6E6",
                zeroline = false,
            ),
            yaxis = attr(
                title = attr(text = "Autoregressive validation MSE", standoff = 12),
                showline = true,
                mirror = true,
                linecolor = "#3A3A3A",
                linewidth = 1,
                ticks = "outside",
                gridcolor = "#E6E6E6",
                zeroline = false,
            ),
            legend = attr(
                x = 0.985,
                y = 0.985,
                xanchor = "right",
                yanchor = "top",
                bgcolor = "rgba(255, 255, 255, 0.92)",
                bordercolor = "#CFCFCF",
                borderwidth = 1,
                font = attr(size = 13),
            ),
            hovermode = "closest",
        ),
    )
    output_path = joinpath(analysis_directory, "validation_pareto_checkpoints.svg")
    PlotlyJS.savefig(plot_handle, output_path; width = 900, height = 560)
    return (plot_handle, output_path)
end

function select_native_closed_loop_candidate(current_front; candidate_id = nothing)
    native_candidates = filter(
        record -> Symbol(archive_value(record, :pareto_scope; default = :default)) === :native &&
                  Symbol(archive_value(record, :threshold_id)) === :native &&
                  Bool(archive_value(record, :loadable; default = false)),
        current_front,
    )
    if !isnothing(candidate_id)
        native_candidates = filter(
            record -> string(archive_value(record, :candidate_id)) == string(candidate_id),
            native_candidates,
        )
    end
    isempty(native_candidates) && error(
        isnothing(candidate_id) ?
        "The retained archive has no loadable native candidate." :
        "The retained archive has no loadable native candidate '$candidate_id'.",
    )
    sort!(
        native_candidates;
        by = record -> (
            Float64(archive_value(record, :validation_matching)),
            Int(archive_value(record, :active_inputs)),
            -Int(archive_value(record, :update)),
        ),
    )
    return first(native_candidates)
end

function resolve_candidate_checkpoint(run_directory::AbstractString, candidate)
    recorded_path = archive_value(candidate, :model_path; default = nothing)
    !isnothing(recorded_path) && isfile(string(recorded_path)) && return abspath(string(recorded_path))
    isnothing(recorded_path) && error(
        "Native candidate $(candidate[:candidate_id]) has no recorded model checkpoint.",
    )
    filename = basename(replace(string(recorded_path), '\\' => '/'))
    local_path = joinpath(run_directory, "candidates", filename)
    isfile(local_path) || error(
        "Native candidate checkpoint is missing at both '$recorded_path' and '$local_path'.",
    )
    return abspath(local_path)
end


function closed_loop_file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function fixed_initial_condition_identifier()
    path = normpath(joinpath(@__DIR__, "..", "..", "RBmodel300.jld2"))
    isfile(path) || error("Fixed-IC checkpoint is missing: $path")
    return (path = abspath(path), identifier = "sha256:$(closed_loop_file_sha256(path))")
end

function closed_loop_cache_tag(material::AbstractString)
    return bytes2hex(SHA.sha256(codeunits(material)))[1:16]
end

function load_closed_loop_cache(
    path::AbstractString;
    controller::Symbol,
    episode_steps::Int,
    expert_identifier::AbstractString,
    initial_condition_identifier::AbstractString,
    candidate_id = nothing,
)
    isfile(path) || return nothing
    loaded = try
        normalize_archive_dict(JLD2.load(path))
    catch error_value
        @warn "Ignoring unreadable closed-loop cache" path exception = error_value
        return nothing
    end
    matching = get(loaded, :schema_version, 0) == P6_CLOSED_LOOP_SCHEMA_VERSION &&
               get(loaded, :complete, false) === true &&
               Symbol(get(loaded, :controller, :unknown)) === controller &&
               Int(get(loaded, :episode_steps, -1)) == episode_steps &&
               string(get(loaded, :expert_identifier, "")) == expert_identifier &&
               string(get(loaded, :initial_condition_identifier, "")) == initial_condition_identifier
    if !isnothing(candidate_id)
        matching &= string(get(loaded, :candidate_id, "")) == string(candidate_id)
    end
    matching || return nothing
    return loaded
end

function save_closed_loop_episode!(
    path::AbstractString,
    episode;
    controller::Symbol,
    episode_steps::Int,
    expert_identifier::AbstractString,
    initial_condition,
    run_id = nothing,
    candidate = nothing,
)
    pareto_atomic_save(
        path;
        schema_version = P6_CLOSED_LOOP_SCHEMA_VERSION,
        complete = true,
        controller,
        episode_steps,
        expert_identifier,
        initial_condition_path = initial_condition.path,
        initial_condition_identifier = initial_condition.identifier,
        run_id,
        candidate_id = isnothing(candidate) ? nothing : string(candidate[:candidate_id]),
        candidate_update = isnothing(candidate) ? nothing : Int(candidate[:update]),
        threshold_id = isnothing(candidate) ? nothing : Symbol(candidate[:threshold_id]),
        active_groups = isnothing(candidate) ? nothing : Int(candidate[:active_groups]),
        active_global_inputs = isnothing(candidate) ? nothing : Int(candidate[:active_inputs]),
        active_global_sensor_locations = isnothing(candidate) ? nothing : Int(candidate[:active_sensor_locations]),
        mean_rewards = episode.mean_rewards,
        global_nusselt = episode.global_nusselt,
        actions = episode.actions,
        total_reward = sum(episode.mean_rewards),
        mean_reward = mean(episode.mean_rewards),
        mean_global_nusselt = mean(episode.global_nusselt),
        created_at = string(Dates.now()),
    )
    return normalize_archive_dict(JLD2.load(path))
end

function with_temporary_environment(callback::Function, overrides::AbstractDict)
    previous = Dict{String, Union{Nothing, String}}(
        string(key) => (haskey(ENV, string(key)) ? ENV[string(key)] : nothing)
        for key in keys(overrides)
    )
    try
        for (key, value) in overrides
            ENV[string(key)] = string(value)
        end
        return callback()
    finally
        for (key, value) in previous
            if isnothing(value)
                pop!(ENV, key, nothing)
            else
                ENV[key] = value
            end
        end
    end
end

function initialize_fixed_closed_loop_runtime!(run_directory, config, analysis_directory)
    if !isdefined(@__MODULE__, :EXPERT_APPRENTICE_PROTOCOL)
        overrides = Dict(
            "DISTILLATION_PROTOCOL" => "fixed",
            "DISTILLATION_SKIP_AUTOLOAD" => "true",
            "DISTILLATION_ALLOW_FRESH_EXPERT" => "false",
            "DISTILLATION_FIXED_EXPERT_PATH" => string(config[:expert_path]),
            "REVISION_RUN_SEED" => string(config[:apprentice_seed]),
            "REVISION_RUN_DIRECTORY" => joinpath(run_directory, "closed_loop_runtime"),
            "DISTILLATION_OUTPUT_DIRECTORY" => joinpath(analysis_directory, "apprentice_outputs"),
        )
        with_temporary_environment(overrides) do
            Base.include(
                @__MODULE__,
                joinpath(
                    @__DIR__,
                    "..",
                    "Expert_Apprentice_Distillation",
                    "Expert_Apprentice.jl",
                ),
            )
        end
    end
    protocol = Base.invokelatest(() -> getfield(@__MODULE__, :EXPERT_APPRENTICE_PROTOCOL))
    protocol === :fixed || error(
        "Closed-loop Fixed-IC inspection requires a fresh Julia process or an already loaded Fixed-IC runtime.",
    )
    expert_metadata = Base.invokelatest(
        () -> getfield(@__MODULE__, :DISTILLATION_EXPERT_METADATA),
    )
    loaded_identifier = string(expert_metadata[:identifier])
    expected_identifier = string(config[:expert_identifier])
    loaded_identifier == expected_identifier || error(
        "Loaded expert '$loaded_identifier' does not match pilot expert '$expected_identifier'.",
    )
    return nothing
end

function normalized_closed_loop_action(action)
    values = Float32.(Array(action))
    if ndims(values) == 3 && size(values, 3) == 1
        values = dropdims(values; dims = 3)
    end
    length(values) == 12 || error("Expected 12 actuator actions, got size $(size(values)).")
    return vec(values)
end

function run_fixed_closed_loop_episode(action_function::Function; episode_steps::Int)
    episode_steps > 0 || throw(ArgumentError("episode_steps must be positive."))
    mean_rewards = Vector{Float64}(undef, episode_steps)
    global_nusselt = Vector{Float64}(undef, episode_steps)
    actions = Matrix{Float32}(undef, episode_steps, 12)

    runtime_env = getfield(@__MODULE__, :env)
    reset_function = getfield(@__MODULE__, :reset!)
    initialize_episode = getfield(@__MODULE__, :generate_random_init)
    nusselt_function = getfield(@__MODULE__, :state_Nu)
    Base.invokelatest(reset_function, runtime_env)
    Base.invokelatest(initialize_episode)
    for step in 1:episode_steps
        action = Base.invokelatest(action_function)
        actions[step, :] .= normalized_closed_loop_action(action)
        Base.invokelatest(runtime_env, action)
        mean_rewards[step] = mean(Float64.(runtime_env.reward))
        global_nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
        isfinite(mean_rewards[step]) || error("Non-finite environment reward at step $step.")
        isfinite(global_nusselt[step]) || error("Non-finite global Nusselt number at step $step.")
    end
    return (; mean_rewards, global_nusselt, actions)
end

function write_closed_loop_csv(path, expert_episode, apprentice_episode)
    expert_rewards = Float64.(expert_episode[:mean_rewards])
    apprentice_rewards = Float64.(apprentice_episode[:mean_rewards])
    expert_nusselt = Float64.(expert_episode[:global_nusselt])
    apprentice_nusselt = Float64.(apprentice_episode[:global_nusselt])
    length(expert_rewards) == length(apprentice_rewards) || error(
        "Expert and apprentice closed-loop episodes have different lengths.",
    )
    open(path, "w") do io
        println(io, "step,expert_mean_reward,apprentice_mean_reward,reward_difference,expert_global_nusselt,apprentice_global_nusselt")
        for step in eachindex(expert_rewards)
            println(io, join((
                step,
                expert_rewards[step],
                apprentice_rewards[step],
                apprentice_rewards[step] - expert_rewards[step],
                expert_nusselt[step],
                apprentice_nusselt[step],
            ), ','))
        end
    end
    return path
end

function plot_closed_loop_rewards(expert_episode, apprentice_episode, candidate, analysis_directory)
    expert_rewards = Float64.(expert_episode[:mean_rewards])
    apprentice_rewards = Float64.(apprentice_episode[:mean_rewards])
    steps = collect(eachindex(expert_rewards))
    traces = PlotlyJS.GenericTrace[
        scatter(
            x = steps,
            y = expert_rewards,
            mode = "lines",
            name = "MAT expert",
            line = attr(color = "#252525", width = 3),
            customdata = Float64.(expert_episode[:global_nusselt]),
            hovertemplate = "Step %{x}<br>Mean reward %{y:.6f}<br>Global Nu %{customdata:.6f}<extra>%{fullData.name}</extra>",
        ),
        scatter(
            x = steps,
            y = apprentice_rewards,
            mode = "lines",
            name = "Native apprentice",
            line = attr(color = "#2166AC", width = 3),
            customdata = Float64.(apprentice_episode[:global_nusselt]),
            hovertemplate = "Step %{x}<br>Mean reward %{y:.6f}<br>Global Nu %{customdata:.6f}<extra>%{fullData.name}</extra>",
        ),
    ]
    plot_handle = Plot(
        traces,
        Layout(
            template = "plotly_white",
            title = attr(
                text = "Fixed-IC closed loop — expert vs apprentice (update $(candidate[:update]))",
                x = 0.5,
                xanchor = "center",
                font = attr(size = 22, color = "#252525"),
            ),
            paper_bgcolor = "white",
            plot_bgcolor = "white",
            width = 900,
            height = 560,
            margin = attr(l = 95, r = 30, t = 80, b = 75),
            font = attr(family = "Arial, sans-serif", size = 15, color = "#303030"),
            xaxis = attr(
                title = attr(text = "Control step", standoff = 12),
                showline = true,
                mirror = true,
                linecolor = "#3A3A3A",
                linewidth = 1,
                ticks = "outside",
                gridcolor = "#E6E6E6",
                zeroline = false,
            ),
            yaxis = attr(
                title = attr(text = "Mean environment reward (higher is better)", standoff = 12),
                showline = true,
                mirror = true,
                linecolor = "#3A3A3A",
                linewidth = 1,
                ticks = "outside",
                gridcolor = "#E6E6E6",
                zeroline = false,
            ),
            legend = attr(
                x = 0.985,
                y = 0.02,
                xanchor = "right",
                yanchor = "bottom",
                bgcolor = "rgba(255, 255, 255, 0.92)",
                bordercolor = "#CFCFCF",
                borderwidth = 1,
                font = attr(size = 13),
            ),
            hovermode = "x unified",
        ),
    )
    output_path = joinpath(analysis_directory, "closed_loop_reward_comparison.svg")
    PlotlyJS.savefig(plot_handle, output_path; width = 900, height = 560)
    return (plot_handle, output_path)
end

function inspect_fixed_pilot_closed_loop(
    run_directory,
    config,
    current_front,
    analysis_directory;
    episode_steps::Int = P6_CLOSED_LOOP_EPISODE_STEPS,
    candidate_id = nothing,
)
    candidate = select_native_closed_loop_candidate(current_front; candidate_id)
    expert_identifier = string(config[:expert_identifier])
    initial_condition = fixed_initial_condition_identifier()
    expert_cache_material = join((
        string(P6_CLOSED_LOOP_SCHEMA_VERSION),
        expert_identifier,
        initial_condition.identifier,
        string(episode_steps),
    ), '|')
    expert_cache_directory = joinpath(P6_FIXED_PILOT_INSPECTION_ROOT, "closed_loop_cache")
    mkpath(expert_cache_directory)
    expert_cache_path = joinpath(
        expert_cache_directory,
        "expert_$(closed_loop_cache_tag(expert_cache_material))_$(episode_steps)_steps.jld2",
    )
    apprentice_cache_path = joinpath(
        analysis_directory,
        "closed_loop_apprentice_$(candidate[:candidate_id])_$(episode_steps)_steps.jld2",
    )

    expert_episode = load_closed_loop_cache(
        expert_cache_path;
        controller = :expert,
        episode_steps,
        expert_identifier,
        initial_condition_identifier = initial_condition.identifier,
    )
    apprentice_episode = load_closed_loop_cache(
        apprentice_cache_path;
        controller = :apprentice,
        episode_steps,
        expert_identifier,
        initial_condition_identifier = initial_condition.identifier,
        candidate_id = candidate[:candidate_id],
    )

    if isnothing(expert_episode) || isnothing(apprentice_episode)
        initialize_fixed_closed_loop_runtime!(run_directory, config, analysis_directory)
    end
    if isnothing(expert_episode)
        println("Generating and caching deterministic Fixed-IC expert episode...")
        action_function = () -> begin
            runtime_rl = getfield(@__MODULE__, :RL)
            runtime_agent = getfield(@__MODULE__, :agent)
            runtime_env = getfield(@__MODULE__, :env)
            Base.invokelatest(runtime_rl.prob, runtime_agent.policy, runtime_env).μ
        end
        episode = Base.invokelatest(
            run_fixed_closed_loop_episode,
            action_function;
            episode_steps,
        )
        expert_episode = save_closed_loop_episode!(
            expert_cache_path,
            episode;
            controller = :expert,
            episode_steps,
            expert_identifier,
            initial_condition,
        )
    end
    if isnothing(apprentice_episode)
        checkpoint_path = resolve_candidate_checkpoint(run_directory, candidate)
        loaded_checkpoint = JLD2.load(checkpoint_path)
        haskey(loaded_checkpoint, "model_payload") || error(
            "Candidate checkpoint has no model_payload: $checkpoint_path",
        )
        candidate_model = loaded_checkpoint["model_payload"]
        runtime_flux = Base.invokelatest(() -> getfield(@__MODULE__, :Flux))
        Base.invokelatest(runtime_flux.testmode!, candidate_model)
        input_mask = Float32.(candidate[:mask])
        println(
            "Generating and caching native apprentice episode from update " *
            "$(candidate[:update])...",
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
        episode = Base.invokelatest(
            run_fixed_closed_loop_episode,
            action_function;
            episode_steps,
        )
        apprentice_episode = save_closed_loop_episode!(
            apprentice_cache_path,
            episode;
            controller = :apprentice,
            episode_steps,
            expert_identifier,
            initial_condition,
            run_id = string(candidate[:run_id]),
            candidate,
        )
    end

    csv_path = write_closed_loop_csv(
        joinpath(analysis_directory, "closed_loop_reward_comparison.csv"),
        expert_episode,
        apprentice_episode,
    )
    plot_handle, plot_path = plot_closed_loop_rewards(
        expert_episode,
        apprentice_episode,
        candidate,
        analysis_directory,
    )
    expert_rewards = Float64.(expert_episode[:mean_rewards])
    apprentice_rewards = Float64.(apprentice_episode[:mean_rewards])
    reward_differences = apprentice_rewards .- expert_rewards
    reward_rmse = sqrt(mean(abs2, reward_differences))
    reward_correlation = cor(expert_rewards, apprentice_rewards)
    maximum_difference_step = argmax(abs.(reward_differences))
    summary_io = IOBuffer()
    println(summary_io, "Deterministic Fixed-IC closed-loop comparison")
    println(summary_io, "  native candidate update: $(candidate[:update])")
    println(summary_io, "  active groups:           $(candidate[:active_groups])")
    println(summary_io, "  global sensor locations: $(candidate[:active_sensor_locations])")
    @printf(summary_io, "  expert total reward:     %.8f\n", sum(expert_rewards))
    @printf(summary_io, "  apprentice total reward: %.8f\n", sum(apprentice_rewards))
    @printf(summary_io, "  total reward difference: %.8f\n", sum(reward_differences))
    @printf(summary_io, "  mean reward difference:  %.8f\n", mean(reward_differences))
    @printf(summary_io, "  reward-curve RMSE:        %.8f\n", reward_rmse)
    @printf(summary_io, "  reward-curve correlation: %.8f\n", reward_correlation)
    @printf(
        summary_io,
        "  maximum absolute difference: %.8f (step %d)\n",
        abs(reward_differences[maximum_difference_step]),
        maximum_difference_step,
    )
    @printf(
        summary_io,
        "  expert mean global Nu:     %.8f\n",
        mean(Float64.(expert_episode[:global_nusselt])),
    )
    @printf(
        summary_io,
        "  apprentice mean global Nu: %.8f\n",
        mean(Float64.(apprentice_episode[:global_nusselt])),
    )
    println(summary_io, "  expert cache:            $expert_cache_path")
    println(summary_io, "  apprentice cache:        $apprentice_cache_path")
    println(summary_io, "  comparison CSV:          $csv_path")
    println(summary_io, "  reward plot:             $plot_path")
    summary_text = String(take!(summary_io))
    println()
    print(summary_text)
    summary_path = joinpath(analysis_directory, "closed_loop_summary.txt")
    open(summary_path, "w") do io
        print(io, summary_text)
    end
    println("  closed-loop summary:     $summary_path")
    return (;
        candidate,
        expert_episode,
        apprentice_episode,
        expert_cache_path,
        apprentice_cache_path,
        csv_path,
        plot_handle,
        plot_path,
        summary_path,
        reward_rmse,
        reward_correlation,
    )
end

function render_fixed_pilot_overview(
    run_directory::AbstractString,
    records,
    summary,
    config,
    current_front,
)
    io = IOBuffer()
    updates = sort!(unique(Int(archive_value(record, :update)) for record in records))
    native_records = sort!(
        filter(record -> Symbol(archive_value(record, :threshold_id)) === :native, records);
        by = record -> Int(archive_value(record, :update)),
    )
    invalid_count = count(record -> !valid_pareto_candidate(record), records)
    checkpoint_directory = joinpath(run_directory, "candidates")
    checkpoint_files = isdir(checkpoint_directory) ? filter(
        path -> endswith(path, ".jld2"),
        readdir(checkpoint_directory; join = true),
    ) : String[]

    println(io, "Package-6 Fixed-IC technical pilot overview")
    println(io, "  run:                     ", basename(run_directory))
    println(io, "  status:                  complete")
    @printf(io, "  runtime:                 %.2f s\n", Float64(summary["elapsed_seconds"]))
    println(io, "  training updates:        ", Int(summary["update_count"]))
    @printf(io, "  final training loss:     %.8e\n", Float64(summary["final_training_loss"]))
    @printf(io, "  regularization strength: %.8g\n", Float64(config[:regularization_strength]))
    println(io, "  evaluation shards:       ", length(updates))
    println(io, "  candidate records:       ", length(records))
    println(io, "  numerical failures:      ", invalid_count)
    println(io, "  retained model files:    ", length(checkpoint_files))
    println(io)

    println(io, "Native GO trajectory")
    println(io, "  update  threshold                 groups  inputs  sensors   autoreg. MSE    teacher MSE")
    println(io, "  ------  ------------------------  ------  ------  -------  --------------  --------------")
    for record in native_records
        print_candidate_row(io, record)
    end
    println(io)

    final_update = last(updates)
    final_records = filter(
        record -> Int(archive_value(record, :update)) == final_update,
        records,
    )
    print_front(io, "Final candidate comparison", final_records)
    println(io)
    print_front(io, "Current retained Pareto archive", current_front)

    return String(take!(io))
end

"""
    inspect_fixed_pilot([run_directory]; closed_loop=true, episode_steps=200,
                        closed_loop_candidate_id=nothing)

Inspect the newest completed Fixed-IC technical pilot, export its candidate
history and Pareto plot, and by default compare the best-matching retained
native apprentice against the MAT expert in one deterministic closed-loop
episode. Passing `run_directory` selects a specific run. Set
`closed_loop=false` for the former lightweight, model-free inspection.
"""
function inspect_fixed_pilot(
    run_directory::AbstractString = latest_fixed_pilot_run_directory();
    closed_loop::Bool = true,
    episode_steps::Int = P6_CLOSED_LOOP_EPISODE_STEPS,
    closed_loop_candidate_id = nothing,
)
    run_directory = abspath(run_directory)
    summary_path = joinpath(run_directory, "pilot_summary.jld2")
    config_path = joinpath(run_directory, "config.jld2")
    manifest_path = joinpath(run_directory, "archive", "archive.jld2")
    for path in (summary_path, config_path, manifest_path)
        isfile(path) || error("Required pilot result is missing: $path")
    end

    summary = JLD2.load(summary_path)
    loaded_config = JLD2.load(config_path)
    manifest = JLD2.load(manifest_path)
    config = normalize_archive_dict(loaded_config["config"])
    files = fixed_pilot_evaluation_files(run_directory)
    records = reduce(
        vcat,
        load_evaluation_records.(files);
        init = Dict{Symbol, Any}[],
    )
    sort!(records; by = pilot_record_sort_key)
    current_front = [normalize_archive_dict(record) for record in manifest["front"]]
    historical_front = pareto_front(records)

    overview = render_fixed_pilot_overview(
        run_directory,
        records,
        summary,
        config,
        current_front,
    )
    print(overview)

    analysis_directory = joinpath(run_directory, "analysis")
    mkpath(analysis_directory)
    overview_path = joinpath(analysis_directory, "overview.txt")
    open(overview_path, "w") do io
        print(io, overview)
    end
    history_path = write_candidate_csv(
        joinpath(analysis_directory, "candidate_history.csv"),
        records,
    )
    historical_front_path = write_candidate_csv(
        joinpath(analysis_directory, "historical_pareto_front.csv"),
        historical_front,
    )
    current_front_path = write_candidate_csv(
        joinpath(analysis_directory, "current_archive_front.csv"),
        current_front,
    )
    plot_handle, plot_path = plot_fixed_pilot_validation(
        records,
        current_front,
        analysis_directory,
    )
    closed_loop_result = closed_loop ? inspect_fixed_pilot_closed_loop(
        run_directory,
        config,
        current_front,
        analysis_directory;
        episode_steps,
        candidate_id = closed_loop_candidate_id,
    ) : nothing

    println()
    println("Written analysis files:")
    println("  $overview_path")
    println("  $history_path")
    println("  $historical_front_path")
    println("  $current_front_path")
    println("  $plot_path")
    return (
        run_directory,
        records,
        historical_front,
        current_front,
        overview_path,
        history_path,
        historical_front_path,
        current_front_path,
        plot_handle,
        plot_path,
        closed_loop = closed_loop_result,
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        inspect_fixed_pilot()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
