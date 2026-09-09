using Dates
using JLD2
using Printf
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "SimpleNNAStudy.jl"))
using .SimpleNNAStudy

const SNN_DISTILLATION_DIRECTORY = joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation")
include(joinpath(SNN_DISTILLATION_DIRECTORY, "ParetoArchive.jl"))

const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "results")
const SNN_TEST_STEPS = 200
const THRESHOLD_COLOR_PALETTE = ("#2166AC", "#92C5DE", "#D6604D", "#67001F")
const QUALITY_THRESHOLD_DASHES = ("dash",)
const QUALITY_THRESHOLD_LINE_COLOR = "#555555"
const SELECTED_CANDIDATE_COLOR = "#F2C14E"
const PARETO_SWEEP_CANDIDATE_COLOR = "#7B3294"
const REPLICATE_SYMBOLS = Dict(1 => "circle", 2 => "diamond", 3 => "square")

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "experiment_id" => nothing,
        "config" => nothing,
        "strengths" => Float64[],
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "poll_seconds" => 60,
        "timeout_seconds" => 14 * 24 * 60 * 60,
        "expected_updates" => SNN_UPDATES,
        "retry_failed" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            println("Usage: analyze_configuration_worker.jl --experiment-id ID --config NAME --strength VALUE [--strength VALUE ...] [--results-dir PATH] [--poll-seconds N] [--timeout-seconds N]")
            return nothing
        elseif argument == "--retry-failed"
            values["retry_failed"] = true
            index += 1
            continue
        elseif argument == "--strength"
            index == length(arguments) && error("Missing value after $argument.")
            push!(values["strengths"], parse(Float64, arguments[index + 1]))
            index += 2
            continue
        end
        index == length(arguments) && error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    isnothing(values["experiment_id"]) && error("--experiment-id is required.")
    isnothing(values["config"]) && error("--config is required.")
    strengths = unique(Float64.(values["strengths"]))
    isempty(strengths) && error("At least one --strength is required.")
    all(strength -> isfinite(strength) && strength > 0, strengths) || error("Strengths must be finite and positive.")
    return (
        experiment_id = normalize_experiment_id(values["experiment_id"]),
        configuration = normalize_configuration(values["config"]),
        strengths,
        results_root = abspath(string(values["results_dir"])),
        poll_seconds = parse(Int, string(values["poll_seconds"])),
        timeout_seconds = parse(Int, string(values["timeout_seconds"])),
        expected_updates = parse(Int, string(values["expected_updates"])),
        retry_failed = Bool(values["retry_failed"]),
    )
end

function wait_for_runs(options)
    jobs = [
        job_for(options.experiment_id, options.configuration, strength, replicate; updates = options.expected_updates)
        for strength in options.strengths for replicate in SNN_REPLICATES
    ]
    deadline = time() + options.timeout_seconds
    seen_nonfailed = Dict(job.id => false for job in jobs)
    while true
        statuses = [(job, load_status(status_path(options.results_root, job))) for job in jobs]
        for (job, status) in statuses
            if isnothing(status) || Symbol(status[:state]) !== :failed
                seen_nonfailed[job.id] = true
            end
        end
        failed = [
            (job, status) for (job, status) in statuses
            if !isnothing(status) && Symbol(status[:state]) === :failed &&
               (!options.retry_failed || seen_nonfailed[job.id])
        ]
        if !isempty(failed)
            details = join(("$(job.id): $(get(status, :error, "unknown error"))" for (job, status) in failed), "\n")
            error("Simple-NNA training failed:\n$details")
        end
        all_complete = all((!isnothing(status) && Symbol(status[:state]) === :complete) for (_, status) in statuses)
        all_complete && return jobs
        time() >= deadline && error("Timed out waiting for Simple-NNA runs $(join((job.id for job in jobs), ", ")).")
        states = join(("$(job.id)=$(isnothing(status) ? "missing" : string(status[:state]))" for (job, status) in statuses), ", ")
        println("Waiting: $states")
        flush(stdout)
        sleep(options.poll_seconds)
    end
end

function retain_successful_threshold_records(records; context::AbstractString)
    by_update = Dict{Int, Vector{Dict{Symbol, Any}}}()
    for record in records
        push!(get!(by_update, Int(record[:update]), Dict{Symbol, Any}[]), record)
    end
    retained = Dict{Symbol, Any}[]
    for update in sort!(collect(keys(by_update)))
        batch = by_update[update]
        native = filter(record -> Symbol(record[:threshold_id]) === :native, batch)
        length(native) == 1 || error("Expected one native candidate at update $update in $context.")
        native_record = only(native)
        native_active_groups = Int(native_record[:active_groups])
        push!(retained, native_record)
        for record in batch
            Symbol(record[:threshold_id]) === :native && continue
            Int(record[:active_groups]) < native_active_groups && push!(retained, record)
        end
    end
    return retained
end

function load_run_records(options, job)
    directory = run_directory(options.results_root, job)
    config_path = joinpath(directory, "config.jld2")
    summary_path = joinpath(directory, "summary.jld2")
    evaluations_path = joinpath(directory, "evaluations.jld2")
    all(isfile, (config_path, summary_path, evaluations_path)) || error(
        "Complete run $(job.id) is missing config, summary, or evaluations.jld2.",
    )
    loaded_config = JLD2.load(config_path)
    config = normalize_archive_dict(loaded_config["config"])
    checks = (
        Symbol(config[:experiment]) === :simple_nna_varying_regularizer_comparison,
        Symbol(config[:protocol]) === :varying,
        Symbol(config[:training_data_split]) === :train,
        Symbol(config[:validation_data_split]) === :validation,
        config[:test_data_used_during_training] == false,
        string(config[:experiment_id]) == options.experiment_id,
        string(config[:configuration]) == job.configuration,
        Symbol(config[:method]) === job.method,
        Bool(config[:group_channels]) == job.group_channels,
        Int(config[:replicate]) == job.replicate,
        Float64(config[:regularization_strength]) == job.regularization_strength,
        Int(config[:master_seed]) == SNN_MASTER_SEED,
        Int(config[:apprentice_seed]) == job.apprentice_seed,
        Int(config[:batch_order_seed]) == job.batch_seed,
        Int(config[:regularized_updates]) == options.expected_updates,
        Int(config[:batch_size]) == SNN_BATCH_SIZE,
        Int(config[:validation_batch_size]) == SNN_VALIDATION_BATCH_SIZE,
        Symbol(config[:apprentice_architecture]) === SNN_APPRENTICE_ARCHITECTURE,
        Float64(config[:nna_scale]) == SNN_NNA_SCALE,
        Int(config[:hidden_size]) == SNN_HIDDEN_SIZE,
        Int(config[:mat_actor_parameter_count]) == SNN_MAT_ACTOR_PARAMETER_COUNT,
        Int(config[:apprentice_parameter_count]) == SNN_PARAMETER_COUNT,
        Symbol(config[:threshold_importance_mode]) === :max_input_l1,
        Int(config[:threshold_minimum_active_groups]) == 1,
    )
    all(checks) || error("Run $(job.id) does not match its requested Simple-NNA configuration.")
    collection = load_evaluation_collection(directory)
    collection.consolidated || error("Run $(job.id) evaluations are not consolidated.")
    collection.config_fingerprint == string(loaded_config["config_fingerprint"]) || error(
        "Run $(job.id) evaluation/config fingerprint mismatch.",
    )
    updates = Int[batch[:update] for batch in collection.batches]
    updates == expected_evaluation_updates(options.expected_updates) || error(
        "Run $(job.id) has incomplete evaluation updates.",
    )
    records = Dict{Symbol, Any}[]
    for batch in collection.batches, candidate in batch[:candidates]
        record = normalize_archive_dict(candidate)
        record[:source_run_directory] = directory
        record[:configuration] = job.configuration
        record[:replicate] = job.replicate
        record[:regularization_strength] = Float64(config[:regularization_strength])
        push!(records, record)
    end
    return retain_successful_threshold_records(records; context = job.id)
end

function quality_membership(configuration, value)
    return join((@sprintf("%.12g", threshold)
                 for threshold in quality_thresholds(configuration) if value <= threshold), ";")
end

function write_csv(path::AbstractString, records, front_ids)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "run_id,replicate,configuration,strength,update,candidate_id,threshold_id,threshold_value,active_groups,active_inputs,validation_matching,pooled_pareto,qualified_quality_thresholds")
        for record in records
            @printf(
                io,
                "%s,%d,%s,%.12g,%d,%s,%s,%.12g,%d,%d,%.17g,%s,%s\n",
                string(record[:run_id]), Int(record[:replicate]), string(record[:configuration]),
                Float64(record[:regularization_strength]), Int(record[:update]),
                string(record[:candidate_id]), string(record[:threshold_id]),
                Float64(record[:threshold_value]), Int(record[:active_groups]),
                Int(record[:active_inputs]), Float64(record[:validation_matching]),
                string(string(record[:candidate_id]) in front_ids),
                quality_membership(record[:configuration], Float64(record[:validation_matching])),
            )
        end
    end
    return path
end

function observed_strengths(records)
    strengths = sort!(unique(Float64(record[:regularization_strength]) for record in records))
    isempty(strengths) && error("Cannot determine used strengths from an empty evaluation set.")
    all(strength -> isfinite(strength) && strength > 0, strengths) || error(
        "Evaluation records contain an invalid regularization strength.",
    )
    return strengths
end

function observed_thresholds(records)
    thresholds = sort!(unique(Float64(record[:threshold_value]) for record in records))
    isempty(thresholds) && error("Cannot determine used thresholds from an empty evaluation set.")
    all(threshold -> isfinite(threshold) && threshold >= 0, thresholds) || error(
        "Evaluation records contain an invalid threshold value.",
    )
    return thresholds
end

threshold_colors(thresholds) = Dict(
    threshold => THRESHOLD_COLOR_PALETTE[mod1(index, length(THRESHOLD_COLOR_PALETTE))]
    for (index, threshold) in enumerate(thresholds)
)

function ensure_plotly_loaded!()
    isdefined(@__MODULE__, :PlotlyJS) || Base.eval(@__MODULE__, :(using PlotlyJS))
    return nothing
end

function make_plot(options, records, pooled_front, selections, output_directory)
    ensure_plotly_loaded!()
    return Base.invokelatest(make_plot_loaded, options, records, pooled_front, selections, output_directory)
end

function make_plot_loaded(options, records, pooled_front, selections, output_directory)
    strengths = observed_strengths(records)
    thresholds = observed_thresholds(records)
    colors = threshold_colors(thresholds)
    traces = PlotlyJS.GenericTrace[]
    shown_thresholds = Set{Float64}()
    for replicate in SNN_REPLICATES, threshold in thresholds
        selected = filter(record -> Int(record[:replicate]) == replicate && Float64(record[:threshold_value]) == threshold, records)
        show_threshold_legend = !isempty(selected) && !(threshold in shown_thresholds)
        show_threshold_legend && push!(shown_thresholds, threshold)
        active_groups = Int.(getindex.(selected, :active_groups))
        active_inputs = Int.(getindex.(selected, :active_inputs))
        push!(traces, PlotlyJS.scatter(
            x = active_groups,
            y = Float64.(getindex.(selected, :validation_matching)),
            mode = "markers",
            name = "τ=$(threshold)",
            legendgroup = "threshold_$(threshold)",
            showlegend = show_threshold_legend,
            marker = PlotlyJS.attr(
                color = colors[threshold],
                symbol = REPLICATE_SYMBOLS[replicate],
                size = 5,
                opacity = 0.38,
                line = PlotlyJS.attr(width = 0),
            ),
            customdata = hcat(
                active_groups,
                active_inputs,
                Int.(getindex.(selected, :update)),
                fill(replicate, length(selected)),
                Float64.(getindex.(selected, :regularization_strength)),
            ),
            hovertemplate = "groups=%{customdata[0]}<br>global inputs=%{customdata[1]}<br>MSE=%{y:.4e}<br>update=%{customdata[2]}<br>replicate=%{customdata[3]}<br>strength=%{customdata[4]:.4g}<extra></extra>",
        ))
    end
    front_active_groups = Int.(getindex.(pooled_front, :active_groups))
    front_active_inputs = Int.(getindex.(pooled_front, :active_inputs))
    push!(traces, PlotlyJS.scatter(
        x = front_active_groups,
        y = Float64.(getindex.(pooled_front, :validation_matching)),
        mode = "lines+markers",
        name = "pooled Pareto front",
        line = PlotlyJS.attr(color = "#111111", width = 2.5),
        marker = PlotlyJS.attr(color = "#111111", size = 7, symbol = "circle-open"),
        customdata = hcat(front_active_groups, front_active_inputs),
        hovertemplate = "groups=%{customdata[0]}<br>global inputs=%{customdata[1]}<br>MSE=%{y:.4e}<extra>pooled Pareto front</extra>",
    ))
    shown_candidate_roles = Set{Symbol}()
    for selection in selections.unique_candidates
        candidate = selection[:candidate]
        selected_thresholds = sort(Float64.(selection[:quality_thresholds]); rev = true)
        is_quality_selection = !isempty(selected_thresholds)
        candidate_role = is_quality_selection ? :selected : :additional
        show_candidate_legend = !(candidate_role in shown_candidate_roles)
        push!(shown_candidate_roles, candidate_role)
        push!(traces, PlotlyJS.scatter(
            x = [Int(candidate[:active_groups])],
            y = [Float64(candidate[:validation_matching])],
            mode = "markers",
            name = is_quality_selection ? "Selected test candidate" : "Additional GR-SC candidates",
            legendgroup = is_quality_selection ? "selected" : "gr_sc_sweep",
            showlegend = show_candidate_legend,
            marker = PlotlyJS.attr(
                color = is_quality_selection ? SELECTED_CANDIDATE_COLOR : PARETO_SWEEP_CANDIDATE_COLOR,
                size = 14,
                symbol = "star", line = PlotlyJS.attr(color = "#111111", width = 1.0),
            ),
        ))
    end
    for replicate in SNN_REPLICATES
        push!(traces, PlotlyJS.scatter(
            x = [NaN],
            y = [NaN],
            mode = "markers",
            name = "Replicate $replicate",
            legendgroup = "replicates",
            marker = PlotlyJS.attr(
                color = "#555555",
                symbol = REPLICATE_SYMBOLS[replicate],
                size = 7,
            ),
            hoverinfo = "skip",
        ))
    end
    layout = PlotlyJS.Layout(
        template = "plotly_white",
        title = "Simple NNA $(options.configuration), λ ∈ {$(join(strengths, ", "))}",
        xaxis = PlotlyJS.attr(title = "Active groups"),
        yaxis = PlotlyJS.attr(title = "Validation expert-action matching (MSE)", type = "log"),
        legend = PlotlyJS.attr(title = PlotlyJS.attr(text = "Threshold")),
        shapes = [PlotlyJS.attr(
            type = "line",
            xref = "paper",
            x0 = 0,
            x1 = 1,
            yref = "y",
            y0 = threshold,
            y1 = threshold,
            line = PlotlyJS.attr(
                color = QUALITY_THRESHOLD_LINE_COLOR, width = 1.3,
                dash = QUALITY_THRESHOLD_DASHES[index],
            ),
        ) for (index, threshold) in enumerate(quality_thresholds(options.configuration))],
    )
    plot = PlotlyJS.Plot(traces, layout)
    paths = String[]
    for extension in ("svg", "pdf")
        path = joinpath(output_directory, "pareto_all_points.$extension")
        PlotlyJS.savefig(plot, path; width = 1050, height = 650)
        push!(paths, path)
    end
    return paths
end

file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function candidate_sort_key(record)
    return (
        Int(record[:active_inputs]),
        Int(record[:active_groups]),
        Float64(record[:validation_matching]),
        Int(record[:update]),
        string(record[:run_id]),
        string(record[:candidate_id]),
    )
end

function select_sparse_test_candidate(pooled_front; quality_threshold = SNN_QUALITY_THRESHOLD)
    qualified = filter(
        record -> Float64(record[:validation_matching]) <= quality_threshold,
        pooled_front,
    )
    isempty(qualified) && return nothing
    return first(sort(qualified; by = candidate_sort_key))
end

function select_quality_candidates(pooled_front, configuration)
    threshold_selections = Dict{Symbol, Any}[]
    unique_candidates = Dict{Symbol, Any}[]
    candidate_indices = Dict{String, Int}()
    quality_threshold = only(quality_thresholds(configuration))
    primary = select_sparse_test_candidate(pooled_front; quality_threshold)
    if isnothing(primary)
        push!(threshold_selections, Dict{Symbol, Any}(
            :quality_threshold => quality_threshold,
            :candidate_id => nothing,
            :candidate_index => nothing,
        ))
        return (; threshold_selections, unique_candidates,
                pareto_sweep_candidate_indices = Int[])
    end
    candidates = if configuration == "gr-sc"
        minimum_groups = Int(primary[:active_groups])
        sort(filter(record ->
            minimum_groups <= Int(record[:active_groups]) <= SNN_GR_SC_PARETO_SWEEP_MAX_ACTIVE_GROUPS,
            pooled_front,
        ); by = record -> (Int(record[:active_groups]), candidate_sort_key(record)))
    else
        [primary]
    end
    primary_id = string(primary[:candidate_id])
    for candidate in candidates
        candidate_id = string(candidate[:candidate_id])
        candidate_index = length(unique_candidates) + 1
        candidate_indices[candidate_id] = candidate_index
        push!(unique_candidates, Dict{Symbol, Any}(
            :candidate => candidate,
            :quality_thresholds => candidate_id == primary_id ? Float64[quality_threshold] : Float64[],
            :selection_role => candidate_id == primary_id ? :quality_threshold : :gr_sc_pareto_sweep,
        ))
    end
    primary_index = candidate_indices[primary_id]
    push!(threshold_selections, Dict{Symbol, Any}(
        :quality_threshold => quality_threshold,
        :candidate_id => primary_id,
        :candidate_index => primary_index,
    ))
    sweep_indices = configuration == "gr-sc" ? collect(eachindex(unique_candidates)) : Int[]
    return (; threshold_selections, unique_candidates,
            pareto_sweep_candidate_indices = sweep_indices)
end

function freeze_test_candidates!(output, selections, configuration)
    clear_selected_candidate_tests!(output)
    frozen_candidates = Dict{Symbol, Any}[]
    for (index, selection) in enumerate(selections.unique_candidates)
        selected = selection[:candidate]
        run_directory = string(selected[:source_run_directory])
        checkpoint_path = candidate_checkpoint_for_record(run_directory, selected)
        candidate = hydrate_candidate_record(selected, run_directory)
        push!(frozen_candidates, Dict{Symbol, Any}(
            :candidate_index => index,
            :quality_thresholds => copy(selection[:quality_thresholds]),
            :selection_role => selection[:selection_role],
            :candidate => candidate,
            :checkpoint_path => abspath(checkpoint_path),
            :checkpoint_sha256 => file_sha256(checkpoint_path),
        ))
    end
    path = joinpath(output, "selected_test_candidates.jld2")
    atomic_save(
        path;
        schema_version = SNN_SCHEMA_VERSION,
        experiment = :simple_nna_varying_regularizer_comparison,
        selection_source = :pooled_validation_pareto_front,
        selection_rule = configuration == "gr-sc" ?
            :quality_threshold_candidate_plus_pareto_sweep_to_max_active_groups :
            :minimum_active_inputs_under_quality_threshold,
        quality_thresholds = collect(quality_thresholds(configuration)),
        threshold_selections = selections.threshold_selections,
        unique_candidate_count = length(frozen_candidates),
        candidates = frozen_candidates,
        pareto_sweep_candidate_indices = selections.pareto_sweep_candidate_indices,
        pareto_sweep_max_active_groups = configuration == "gr-sc" ?
            SNN_GR_SC_PARETO_SWEEP_MAX_ACTIVE_GROUPS : nothing,
        selection_uses_test_data = false,
        frozen_before_test = true,
        frozen_at = string(Dates.now()),
    )
    return (; path, candidates = frozen_candidates,
            threshold_selections = selections.threshold_selections)
end

function configure_test_runtime!(candidate, output)
    run_directory = string(candidate[:source_run_directory])
    config = normalize_archive_dict(JLD2.load(joinpath(run_directory, "config.jld2"))["config"])
    expert_path = string(config[:expert_path])
    ENV["DISTILLATION_PROTOCOL"] = "varying"
    ENV["DISTILLATION_APPRENTICE_ARCHITECTURE"] = "simple_nna"
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(Bool(config[:group_channels]))
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV["DISTILLATION_VARYING_EXPERT_PATH"] = expert_path
    ENV["REVISION_RUN_SEED"] = string(config[:apprentice_seed])
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output, "test", "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output, "test", "apprentice_output")
    Base.include(@__MODULE__, joinpath(SNN_DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
    expert_metadata = Base.invokelatest(() -> getfield(@__MODULE__, :DISTILLATION_EXPERT_METADATA))
    string(expert_metadata[:identifier]) == string(config[:expert_identifier]) || error(
        "Loaded Varying expert does not match the selected candidate's training corpus.",
    )
    return config
end


function normalize_test_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 && (values = dropdims(values; dims = 3))
    length(values) == 12 || error("Expected twelve actuator actions, got $(size(values)).")
    return vec(values)
end

function varying_test_cases()
    corpus = Base.invokelatest(() -> getfield(@__MODULE__, :CORPUS))
    base_seeds = sort!(Int.(collect(keys(corpus[:test]))))
    length(base_seeds) == 2 || error("Expected two Varying test bases, found $(length(base_seeds)).")
    choices = vec([
        (split = :test, base_seed, mirror, offset)
        for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)
    ])
    return [merge(choice, (
        episode = index,
        evaluation_seed = SNN_MASTER_SEED + index,
        case_id = "base_$(choice.base_seed)_mirror_$(choice.mirror ? 1 : 0)_offset_$(choice.offset)",
    )) for (index, choice) in enumerate(choices)]
end

function run_masked_test_episode(candidate_model, input_mask, choice)
    runtime_rl = Base.invokelatest(() -> getfield(@__MODULE__, :RL))
    runtime_env = Base.invokelatest(() -> getfield(@__MODULE__, :env))
    initialize_episode = Base.invokelatest(() -> getfield(@__MODULE__, :generate_random_init))
    nusselt_function = Base.invokelatest(() -> getfield(@__MODULE__, :state_Nu))
    Random.seed!(choice.evaluation_seed)
    Base.invokelatest(
        initialize_episode;
        split = choice.split,
        base_seed = choice.base_seed,
        mirror = choice.mirror,
        offset = choice.offset,
    )
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    rewards = Vector{Float64}(undef, SNN_TEST_STEPS)
    state_nusselt = Vector{Float64}(undef, SNN_TEST_STEPS)
    simulation_times = Vector{Float64}(undef, SNN_TEST_STEPS)
    actions = Matrix{Float32}(undef, SNN_TEST_STEPS, 12)
    for step in 1:SNN_TEST_STEPS
        action = Base.invokelatest(
            runtime_rl.prob,
            candidate_model,
            runtime_env.state .* input_mask,
            nothing,
        ).μ[:, :, 1]
        actions[step, :] .= normalize_test_action(action)
        Base.invokelatest(runtime_env, action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        state_nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
        simulation = Base.invokelatest(() -> getfield(@__MODULE__, :simulation))
        simulation_times[step] = Float64(simulation.model.clock.time)
        isfinite(rewards[step]) && isfinite(state_nusselt[step]) || error(
            "Non-finite Simple-NNA test value in $(choice.case_id), step $step.",
        )
    end
    return (;
        case_id = choice.case_id,
        split = choice.split,
        base_seed = choice.base_seed,
        mirror = choice.mirror,
        offset = choice.offset,
        evaluation_seed = choice.evaluation_seed,
        episode = choice.episode,
        rewards,
        state_nusselt,
        simulation_times,
        actions,
        reward_sum = sum(rewards),
        mean_reward = mean(rewards),
        sum_state_nusselt = sum(state_nusselt),
        negative_sum_state_nusselt = -sum(state_nusselt),
        mean_state_nusselt = mean(state_nusselt),
    )
end

function write_test_csv(path, episodes)
    open(path, "w") do io
        println(io, "case,split,base_seed,mirror,offset,evaluation_seed,episode,step,simulation_time,reward,state_nusselt")
        for result in episodes, step in 1:SNN_TEST_STEPS
            println(io, join((
                result.case_id, result.split, result.base_seed, result.mirror,
                result.offset, result.evaluation_seed, result.episode, step,
                result.simulation_times[step], result.rewards[step], result.state_nusselt[step],
            ), ','))
        end
    end
    return path
end

function make_test_plot(output, episodes, candidate, selected_quality_thresholds)
    ensure_plotly_loaded!()
    return Base.invokelatest(
        make_test_plot_loaded, output, episodes, candidate, selected_quality_thresholds,
    )
end

function make_test_plot_loaded(output, episodes, candidate, selected_quality_thresholds)
    plot = PlotlyJS.make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = reshape(["Environment reward", "Full-state Nu"], :, 1),
    )
    steps = collect(1:SNN_TEST_STEPS)
    for result in episodes
        PlotlyJS.add_trace!(plot, PlotlyJS.scatter(
            x = steps, y = result.rewards, mode = "lines", name = result.case_id,
            legendgroup = result.case_id, showlegend = true,
            line = PlotlyJS.attr(color = "#277DA1", width = 1.5), opacity = 0.65,
        ); row = 1, col = 1)
        PlotlyJS.add_trace!(plot, PlotlyJS.scatter(
            x = steps, y = result.state_nusselt, mode = "lines", name = result.case_id,
            legendgroup = result.case_id, showlegend = false,
            line = PlotlyJS.attr(color = "#F2A13A", width = 1.5), opacity = 0.65,
        ); row = 1, col = 2)
    end
    selection_label = isempty(selected_quality_thresholds) ?
        "GR-SC Pareto sweep: $(candidate[:active_groups]) active groups" :
        "q ∈ {$(join(selected_quality_thresholds, ", "))}"
    PlotlyJS.relayout!(plot, Dict{Symbol, Any}(
        :template => "plotly_white",
        :title => "Masked Simple-NNA Varying test: $(candidate[:active_inputs]) active inputs, $selection_label",
        :width => 1100,
        :height => 500,
        :xaxis => PlotlyJS.attr(title = "Control step"),
        :xaxis2 => PlotlyJS.attr(title = "Control step"),
        :yaxis => PlotlyJS.attr(title = "Mean reward"),
        :yaxis2 => PlotlyJS.attr(title = "Nu"),
    ))
    path = joinpath(output, "test_curves.svg")
    PlotlyJS.savefig(plot, path; width = 1100, height = 500)
    return path
end

function run_frozen_candidate_test!(output, selection_path, frozen, config, cases)
    index = Int(frozen[:candidate_index])
    candidate = frozen[:candidate]
    checkpoint_path = string(frozen[:checkpoint_path])
    selected_quality_thresholds = Float64.(frozen[:quality_thresholds])
    file_sha256(checkpoint_path) == string(frozen[:checkpoint_sha256]) || error(
        "Frozen candidate checkpoint changed before test: $checkpoint_path",
    )
    test_directory = joinpath(output, "test", @sprintf("candidate_%02d", index))
    mkpath(test_directory)
    checkpoint = JLD2.load(checkpoint_path)
    haskey(checkpoint, "model_payload") || error("Candidate checkpoint has no model_payload: $checkpoint_path")
    candidate_model = checkpoint["model_payload"]
    runtime_flux = Base.invokelatest(() -> getfield(@__MODULE__, :Flux))
    Base.invokelatest(runtime_flux.testmode!, candidate_model)
    input_mask = Float32.(candidate[:mask])
    runtime_env = Base.invokelatest(() -> getfield(@__MODULE__, :env))
    length(input_mask) == size(runtime_env.state, 1) || error("Selected candidate mask has the wrong length.")
    episodes = [Base.invokelatest(run_masked_test_episode, candidate_model, input_mask, choice) for choice in cases]
    csv_path = write_test_csv(joinpath(test_directory, "test_episodes.csv"), episodes)
    plot_path = make_test_plot(test_directory, episodes, candidate, selected_quality_thresholds)
    result_path = joinpath(test_directory, "test_results.jld2")
    atomic_save(
        result_path;
        schema_version = SNN_SCHEMA_VERSION,
        experiment = :simple_nna_masked_apprentice_test,
        protocol = :varying,
        selection_uses_test_data = false,
        selection_path,
        candidate_index = index,
        quality_thresholds = selected_quality_thresholds,
        selection_role = Symbol(frozen[:selection_role]),
        candidate_id = string(candidate[:candidate_id]),
        run_id = string(candidate[:run_id]),
        configuration = string(candidate[:configuration]),
        regularization_strength = Float64(candidate[:regularization_strength]),
        replicate = Int(candidate[:replicate]),
        update = Int(candidate[:update]),
        threshold_id = Symbol(candidate[:threshold_id]),
        threshold_value = Float64(candidate[:threshold_value]),
        validation_matching = Float64(candidate[:validation_matching]),
        active_groups = Int(candidate[:active_groups]),
        active_inputs = Int(candidate[:active_inputs]),
        input_mask,
        checkpoint_path,
        checkpoint_sha256 = file_sha256(checkpoint_path),
        expert_identifier = string(config[:expert_identifier]),
        steps = SNN_TEST_STEPS,
        case_count = length(episodes),
        cases,
        episodes,
        reward_sums = Float64[result.reward_sum for result in episodes],
        mean_reward = mean(result.mean_reward for result in episodes),
        sum_state_nusselt = Float64[result.sum_state_nusselt for result in episodes],
        mean_state_nusselt = mean(result.mean_state_nusselt for result in episodes),
        csv_path,
        plot_path,
        completed_at = string(Dates.now()),
    )
    return (; index, candidate, quality_thresholds = selected_quality_thresholds,
            selection_path, result_path, csv_path, plot_path)
end

function clear_selected_candidate_tests!(output)
    for path in (
        joinpath(output, "selected_test_candidate.jld2"),
        joinpath(output, "selected_test_candidates.jld2"),
        joinpath(output, "test", "test_results.jld2"),
        joinpath(output, "test", "test_episode.csv"),
        joinpath(output, "test", "test_episodes.csv"),
        joinpath(output, "test", "test_curves.svg"),
    )
        isfile(path) && rm(path; force = true)
    end
    test_root = joinpath(output, "test")
    if isdir(test_root)
        for entry in readdir(test_root; join = true)
            isdir(entry) && startswith(basename(entry), "candidate_") &&
                rm(entry; recursive = true, force = true)
        end
    end
    return nothing
end

clear_selected_candidate_test!(output) = clear_selected_candidate_tests!(output)

function run_selected_candidate_tests!(output, selections, configuration)
    frozen = freeze_test_candidates!(output, selections, configuration)
    isempty(frozen.candidates) && return (; frozen, tests = NamedTuple[])
    config = configure_test_runtime!(frozen.candidates[1][:candidate], output)
    cases = varying_test_cases()
    tests = [
        run_frozen_candidate_test!(output, frozen.path, candidate, config, cases)
        for candidate in frozen.candidates
    ]
    return (; frozen, tests)
end

function analyze_completed_runs(options, jobs)
    output = analysis_directory(options.results_root, options.experiment_id, options.configuration)
    mkpath(output)
    records = reduce(vcat, (load_run_records(options, job) for job in jobs); init = Dict{Symbol, Any}[])
    strengths = observed_strengths(records)
    expected_native_count = length(jobs) * length(expected_evaluation_updates(options.expected_updates))
    native_count = count(record -> Symbol(record[:threshold_id]) === :native, records)
    native_count == expected_native_count || error(
        "Expected $expected_native_count native evaluation points, found $native_count.",
    )
    pooled_front = pareto_front(records)
    front_ids = Set(string(record[:candidate_id]) for record in pooled_front)
    csv_path = write_csv(joinpath(output, "evaluations.csv"), records, front_ids)
    legacy_csv_path = joinpath(output, "pareto_points.csv")
    isfile(legacy_csv_path) && rm(legacy_csv_path; force = true)
    front_csv_path = write_csv(
        joinpath(output, "pooled_pareto_front.csv"),
        pooled_front,
        front_ids,
    )
    selections = select_quality_candidates(pooled_front, options.configuration)
    plot_paths = make_plot(options, records, pooled_front, selections, output)
    test = run_selected_candidate_tests!(output, selections, options.configuration)
    selected_ids = [string(entry[:candidate][:candidate_id]) for entry in test.frozen.candidates]
    legacy_data_path = joinpath(output, "pareto_points.jld2")
    isfile(legacy_data_path) && rm(legacy_data_path; force = true)
    write_status!(
        analysis_status_path(options.results_root, options.experiment_id, options.configuration);
        state = :complete,
        experiment_id = options.experiment_id,
        configuration = options.configuration,
        regularization_strengths = strengths,
        point_count = length(records),
        front_count = length(pooled_front),
        csv_path,
        front_csv_path,
        plot_paths,
        quality_thresholds = collect(quality_thresholds(options.configuration)),
        threshold_selections = selections.threshold_selections,
        pareto_sweep_candidate_indices = selections.pareto_sweep_candidate_indices,
        pareto_sweep_max_active_groups = options.configuration == "gr-sc" ?
            SNN_GR_SC_PARETO_SWEEP_MAX_ACTIVE_GROUPS : nothing,
        selected_test_candidates = selected_ids,
        selection_path = test.frozen.path,
        test_result_paths = [entry.result_path for entry in test.tests],
        test_csv_paths = [entry.csv_path for entry in test.tests],
        test_plot_paths = [entry.plot_path for entry in test.tests],
        completed_at = string(Dates.now()),
    )
    println("Completed Simple-NNA Pareto analysis for $(options.configuration), λ ∈ {$(join(strengths, ", "))}.")
    println("  points/front: $(length(records)) / $(length(pooled_front))")
    println("  unique selected test candidates: $(length(test.tests))")
    println("  output: $output")
    return (; records, pooled_front, selections, csv_path, front_csv_path, plot_paths, test)
end

function analysis_main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    status = analysis_status_path(options.results_root, options.experiment_id, options.configuration)
    try
        write_status!(status; state = :waiting, experiment_id = options.experiment_id,
                      configuration = options.configuration,
                      regularization_strengths = options.strengths, started_at = string(Dates.now()))
        jobs = wait_for_runs(options)
        return analyze_completed_runs(options, jobs)
    catch exception
        write_status!(status; state = :failed, experiment_id = options.experiment_id,
                      configuration = options.configuration,
                      regularization_strengths = options.strengths,
                      error = sprint(showerror, exception), failed_at = string(Dates.now()))
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    analysis_main()
end
