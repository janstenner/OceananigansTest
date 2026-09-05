using Dates
using JLD2
using Printf
using Random
using SHA
using Statistics

include(joinpath(@__DIR__, "HigherRaGOStudy.jl"))
using .HigherRaGOStudy

const DISTILLATION_DIRECTORY = normpath(joinpath(@__DIR__, "..", "..", "Expert_Apprentice_Distillation"))
include(joinpath(DISTILLATION_DIRECTORY, "ParetoArchive.jl"))

const TEST_STEPS = 200
const MASK_THRESHOLD_COLORS = ("#2166AC", "#92C5DE", "#D6604D", "#67001F")
const QUALITY_THRESHOLD_COLORS = ("#555555", "#8C510A", "#B2182B")
const REPLICATE_SYMBOLS = Dict(1 => "circle", 2 => "diamond", 3 => "square")

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "study" => nothing,
        "experiment_id" => nothing,
        "config" => nothing,
        "strengths" => Float64[],
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "poll_seconds" => 60,
        "timeout_seconds" => 14 * 24 * 60 * 60,
        "expected_updates" => HR_UPDATES,
        "retry_failed" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            println("Usage: analyze_configuration_worker.jl --study ra5e4|ra1e5 --experiment-id ID --config NAME --strength VALUE [--strength VALUE ...] [--results-dir PATH] [--poll-seconds N] [--timeout-seconds N]")
            return nothing
        elseif argument == "--retry-failed"
            values["retry_failed"] = true
            index += 1
        elseif argument == "--strength"
            index < length(arguments) || error("Missing value after $argument.")
            push!(values["strengths"], parse(Float64, arguments[index + 1]))
            index += 2
        elseif startswith(argument, "--")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(values, key) || error("Unknown option '$argument'.")
            values[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument '$argument'.")
        end
    end
    for key in ("study", "experiment_id", "config")
        isnothing(values[key]) && error("--$(replace(key, "_" => "-")) is required.")
    end
    strengths = unique(Float64.(values["strengths"]))
    isempty(strengths) && error("At least one --strength is required.")
    all(strength -> isfinite(strength) && strength > 0, strengths) || error(
        "Strengths must be finite and positive.",
    )
    return (;
        study = normalize_study(values["study"]),
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
        job_for(options.study, options.experiment_id, options.configuration,
                strength, replicate; updates = options.expected_updates)
        for strength in options.strengths for replicate in HR_REPLICATES
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
            details = join(("$(job.id): $(get(status, :error, "unknown error"))"
                            for (job, status) in failed), "\n")
            error("Higher-Ra training failed:\n$details")
        end
        all((!isnothing(status) && Symbol(status[:state]) === :complete)
            for (_, status) in statuses) && return jobs
        time() >= deadline && error(
            "Timed out waiting for Higher-Ra runs $(join((job.id for job in jobs), ", ")).",
        )
        states = join(("$(job.id)=$(isnothing(status) ? "missing" : string(status[:state]))"
                       for (job, status) in statuses), ", ")
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
        Symbol(config[:experiment]) === :higher_ra_go_gr_comparison,
        Symbol(config[:study]) === options.study,
        Symbol(config[:protocol]) === study(options.study).protocol,
        Float64(config[:rayleigh]) == study(options.study).rayleigh,
        Symbol(config[:training_data_split]) === :train,
        Symbol(config[:validation_data_split]) === :validation,
        config[:test_data_used_during_training] == false,
        string(config[:experiment_id]) == options.experiment_id,
        string(config[:configuration]) == job.configuration,
        Symbol(config[:method]) === job.method,
        Bool(config[:group_channels]) == job.group_channels,
        Int(config[:replicate]) == job.replicate,
        Float64(config[:regularization_strength]) == job.regularization_strength,
        Int(config[:master_seed]) == HR_MASTER_SEED,
        Int(config[:apprentice_seed]) == job.apprentice_seed,
        Int(config[:batch_order_seed]) == job.batch_seed,
        Int(config[:regularized_updates]) == options.expected_updates,
        Int(config[:batch_size]) == HR_BATCH_SIZE,
        Int(config[:validation_batch_size]) == HR_VALIDATION_BATCH_SIZE,
        Float64.(config[:quality_threshold_values]) == collect(HR_QUALITY_THRESHOLDS),
        Symbol(config[:threshold_importance_mode]) === :max_input_l1,
        Int(config[:threshold_minimum_active_groups]) == 1,
    )
    all(checks) || error("Run $(job.id) does not match its Higher-Ra configuration.")
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
        record[:study] = options.study
        record[:configuration] = job.configuration
        record[:replicate] = job.replicate
        record[:regularization_strength] = Float64(config[:regularization_strength])
        push!(records, record)
    end
    return retain_successful_threshold_records(records; context = job.id)
end

function quality_membership(value)
    return join((@sprintf("%.12g", threshold)
                 for threshold in HR_QUALITY_THRESHOLDS if value <= threshold), ";")
end

function write_csv(path::AbstractString, records, front_ids)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "run_id,replicate,study,configuration,strength,update,candidate_id,threshold_id,threshold_value,active_groups,active_inputs,validation_matching,pooled_pareto,qualified_quality_thresholds")
        for record in records
            matching = Float64(record[:validation_matching])
            @printf(
                io,
                "%s,%d,%s,%s,%.12g,%d,%s,%s,%.12g,%d,%d,%.17g,%s,%s\n",
                string(record[:run_id]), Int(record[:replicate]), string(record[:study]),
                string(record[:configuration]), Float64(record[:regularization_strength]),
                Int(record[:update]), string(record[:candidate_id]),
                string(record[:threshold_id]), Float64(record[:threshold_value]),
                Int(record[:active_groups]), Int(record[:active_inputs]), matching,
                string(string(record[:candidate_id]) in front_ids), quality_membership(matching),
            )
        end
    end
    return path
end

observed_strengths(records) = sort!(unique(
    Float64(record[:regularization_strength]) for record in records
))

observed_mask_thresholds(records) = sort!(unique(
    Float64(record[:threshold_value]) for record in records
))

mask_threshold_colors(thresholds) = Dict(
    threshold => MASK_THRESHOLD_COLORS[mod1(index, length(MASK_THRESHOLD_COLORS))]
    for (index, threshold) in enumerate(thresholds)
)

function ensure_plotly_loaded!()
    isdefined(@__MODULE__, :PlotlyJS) || Base.eval(@__MODULE__, :(using PlotlyJS))
    return nothing
end

function make_pareto_plot(options, records, pooled_front, output_directory)
    ensure_plotly_loaded!()
    return Base.invokelatest(
        make_pareto_plot_loaded,
        options,
        records,
        pooled_front,
        output_directory,
    )
end

function make_pareto_plot_loaded(options, records, pooled_front, output_directory)
    strengths = observed_strengths(records)
    thresholds = observed_mask_thresholds(records)
    colors = mask_threshold_colors(thresholds)
    traces = PlotlyJS.GenericTrace[]
    shown_thresholds = Set{Float64}()
    for replicate in HR_REPLICATES, threshold in thresholds
        selected = filter(
            record -> Int(record[:replicate]) == replicate &&
                      Float64(record[:threshold_value]) == threshold,
            records,
        )
        show_threshold_legend = !isempty(selected) && !(threshold in shown_thresholds)
        show_threshold_legend && push!(shown_thresholds, threshold)
        active_groups = Int.(getindex.(selected, :active_groups))
        active_inputs = Int.(getindex.(selected, :active_inputs))
        push!(traces, PlotlyJS.scatter(
            x = active_groups,
            y = Float64.(getindex.(selected, :validation_matching)),
            mode = "markers",
            name = "mask τ=$(threshold)",
            legendgroup = "mask_threshold_$(threshold)",
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
    front_groups = Int.(getindex.(pooled_front, :active_groups))
    front_inputs = Int.(getindex.(pooled_front, :active_inputs))
    push!(traces, PlotlyJS.scatter(
        x = front_groups,
        y = Float64.(getindex.(pooled_front, :validation_matching)),
        mode = "lines+markers",
        name = "pooled Pareto front",
        line = PlotlyJS.attr(color = "#111111", width = 2.5),
        marker = PlotlyJS.attr(color = "#111111", size = 7, symbol = "circle-open"),
        customdata = hcat(front_groups, front_inputs),
        hovertemplate = "groups=%{customdata[0]}<br>global inputs=%{customdata[1]}<br>MSE=%{y:.4e}<extra>pooled Pareto front</extra>",
    ))
    for replicate in HR_REPLICATES
        push!(traces, PlotlyJS.scatter(
            x = [NaN], y = [NaN], mode = "markers", name = "Replicate $replicate",
            legendgroup = "replicates",
            marker = PlotlyJS.attr(color = "#555555",
                                   symbol = REPLICATE_SYMBOLS[replicate], size = 7),
            hoverinfo = "skip",
        ))
    end
    shapes = [PlotlyJS.attr(
        type = "line", xref = "paper", x0 = 0, x1 = 1, yref = "y",
        y0 = threshold, y1 = threshold,
        line = PlotlyJS.attr(color = QUALITY_THRESHOLD_COLORS[index],
                             width = 1.5, dash = "dash"),
    ) for (index, threshold) in enumerate(HR_QUALITY_THRESHOLDS)]
    plot = PlotlyJS.Plot(traces, PlotlyJS.Layout(
        template = "plotly_white",
        title = "$(study(options.study).label) $(options.configuration), λ ∈ {$(join(strengths, ", "))}",
        xaxis = PlotlyJS.attr(title = "Active groups"),
        yaxis = PlotlyJS.attr(title = "Validation expert-action matching (MSE)", type = "log"),
        legend = PlotlyJS.attr(title = PlotlyJS.attr(text = "Mask threshold")),
        shapes = shapes,
    ))
    paths = String[]
    for extension in ("svg", "pdf")
        path = joinpath(output_directory, "pareto_all_points.$extension")
        PlotlyJS.savefig(plot, path; width = 1050, height = 650)
        push!(paths, path)
    end
    return paths
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

function select_quality_candidates(pooled_front)
    threshold_selections = Dict{Symbol, Any}[]
    unique_candidates = Dict{Symbol, Any}[]
    candidate_indices = Dict{String, Int}()
    for quality_threshold in HR_QUALITY_THRESHOLDS
        qualified = filter(
            record -> Float64(record[:validation_matching]) <= quality_threshold,
            pooled_front,
        )
        if isempty(qualified)
            push!(threshold_selections, Dict{Symbol, Any}(
                :quality_threshold => quality_threshold,
                :candidate_id => nothing,
                :candidate_index => nothing,
            ))
            continue
        end
        candidate = first(sort(qualified; by = candidate_sort_key))
        candidate_id = string(candidate[:candidate_id])
        candidate_index = get(candidate_indices, candidate_id, 0)
        if candidate_index == 0
            candidate_index = length(unique_candidates) + 1
            candidate_indices[candidate_id] = candidate_index
            push!(unique_candidates, Dict{Symbol, Any}(
                :candidate => candidate,
                :quality_thresholds => Float64[quality_threshold],
            ))
        else
            push!(unique_candidates[candidate_index][:quality_thresholds], quality_threshold)
        end
        push!(threshold_selections, Dict{Symbol, Any}(
            :quality_threshold => quality_threshold,
            :candidate_id => candidate_id,
            :candidate_index => candidate_index,
        ))
    end
    length(unique_candidates) <= length(HR_QUALITY_THRESHOLDS) || error(
        "Quality-threshold selection produced too many candidates.",
    )
    return (; threshold_selections, unique_candidates)
end

function clear_test_outputs!(output)
    selection_path = joinpath(output, "selected_test_candidates.jld2")
    isfile(selection_path) && rm(selection_path; force = true)
    for index in 1:length(HR_QUALITY_THRESHOLDS)
        directory = joinpath(output, "test", @sprintf("candidate_%02d", index))
        isdir(directory) && rm(directory; recursive = true, force = true)
    end
    return nothing
end

function freeze_test_candidates!(output, selections)
    clear_test_outputs!(output)
    frozen_candidates = Dict{Symbol, Any}[]
    for (index, selection) in enumerate(selections.unique_candidates)
        record = selection[:candidate]
        run_directory = string(record[:source_run_directory])
        candidate = hydrate_candidate_record(record, run_directory)
        checkpoint_path = candidate_checkpoint_for_record(run_directory, candidate)
        push!(frozen_candidates, Dict{Symbol, Any}(
            :candidate_index => index,
            :quality_thresholds => copy(selection[:quality_thresholds]),
            :candidate => candidate,
            :checkpoint_path => abspath(checkpoint_path),
            :checkpoint_sha256 => file_sha256(checkpoint_path),
        ))
    end
    path = joinpath(output, "selected_test_candidates.jld2")
    atomic_save(
        path;
        schema_version = HR_SCHEMA_VERSION,
        experiment = :higher_ra_go_gr_comparison,
        selection_source = :pooled_validation_pareto_front,
        selection_rule = :minimum_active_inputs_under_each_quality_threshold,
        quality_thresholds = collect(HR_QUALITY_THRESHOLDS),
        threshold_selections = selections.threshold_selections,
        unique_candidate_count = length(frozen_candidates),
        candidates = frozen_candidates,
        selection_uses_test_data = false,
        frozen_before_test = true,
        frozen_at = string(Dates.now()),
    )
    return (; path, candidates = frozen_candidates,
            threshold_selections = selections.threshold_selections)
end

function configure_test_runtime!(options, frozen_candidates, output)
    isempty(frozen_candidates) && error("Cannot configure test runtime without candidates.")
    study_config = study(options.study)
    current_sources = source_manifest(options.study)
    first_candidate = frozen_candidates[1][:candidate]
    run_directory = string(first_candidate[:source_run_directory])
    config = normalize_archive_dict(JLD2.load(joinpath(run_directory, "config.jld2"))["config"])
    for frozen in frozen_candidates
        candidate = frozen[:candidate]
        candidate_config = normalize_archive_dict(JLD2.load(joinpath(
            string(candidate[:source_run_directory]),
            "config.jld2",
        ))["config"])
        checks = (
            Symbol(candidate_config[:study]) === options.study,
            string(candidate_config[:configuration]) == options.configuration,
            Bool(candidate_config[:group_channels]) == configuration(options.configuration).group_channels,
            string(candidate_config[:expert_sha256]) == current_sources.expert_sha256,
            string(candidate_config[:run_file_sha256]) == current_sources.run_file_sha256,
            string(candidate_config[:state_corpus_sha256]) == current_sources.state_corpus_sha256,
        )
        all(checks) || error(
            "Selected candidate $(candidate[:candidate_id]) has stale or mismatched provenance.",
        )
    end
    ENV["DISTILLATION_PROTOCOL"] = "varying"
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "false"
    ENV["DISTILLATION_AUTOLOAD_PROTOCOL"] = "varying"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(Bool(config[:group_channels]))
    ENV["DISTILLATION_RUN_FILE"] = study_config.run_file
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV["DISTILLATION_VARYING_EXPERT_PATH"] = study_config.expert
    ENV["DISTILLATION_WORKER_DIRECTORY"] = joinpath(study_config.distillation_root, "varying")
    ENV["REVISION_RUN_SEED"] = string(config[:apprentice_seed])
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output, "test", "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output, "test", "apprentice_output")
    Base.include(@__MODULE__, joinpath(DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
    expert_metadata = Base.invokelatest(() -> getfield(@__MODULE__, :DISTILLATION_EXPERT_METADATA))
    string(expert_metadata[:identifier]) == string(config[:expert_identifier]) || error(
        "Loaded Higher-Ra expert does not match the selected candidate's corpus.",
    )
    Float64(Base.invokelatest(() -> getfield(@__MODULE__, :RA))) == study_config.rayleigh || error(
        "Test runtime loaded the wrong Rayleigh number.",
    )
    return config
end

function normalize_test_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 && (values = dropdims(values; dims = 3))
    length(values) == 12 || error("Expected twelve actuator actions, got $(size(values)).")
    return vec(values)
end

function higher_ra_test_cases()
    corpus = Base.invokelatest(() -> getfield(@__MODULE__, :CORPUS))
    test_split = haskey(corpus, :test) ? corpus[:test] : corpus["test"]
    base_seeds = sort!(Int.(collect(keys(test_split))))
    length(base_seeds) == 2 || error("Expected two Higher-Ra test bases.")
    choices = vec([
        (split = :test, base_seed, mirror, offset)
        for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)
    ])
    return [merge(choice, (
        episode = index,
        evaluation_seed = HR_MASTER_SEED + index,
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
    rewards = Vector{Float64}(undef, TEST_STEPS)
    state_nusselt = Vector{Float64}(undef, TEST_STEPS)
    simulation_times = Vector{Float64}(undef, TEST_STEPS)
    actions = Matrix{Float32}(undef, TEST_STEPS, 12)
    for step in 1:TEST_STEPS
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
            "Non-finite Higher-Ra test value in $(choice.case_id), step $step.",
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
        for result in episodes, step in 1:TEST_STEPS
            println(io, join((
                result.case_id, result.split, result.base_seed, result.mirror,
                result.offset, result.evaluation_seed, result.episode, step,
                result.simulation_times[step], result.rewards[step],
                result.state_nusselt[step],
            ), ','))
        end
    end
    return path
end

function make_test_plot(directory, episodes, candidate, quality_thresholds)
    ensure_plotly_loaded!()
    return Base.invokelatest(
        make_test_plot_loaded,
        directory,
        episodes,
        candidate,
        quality_thresholds,
    )
end

function make_test_plot_loaded(directory, episodes, candidate, quality_thresholds)
    plot = PlotlyJS.make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = reshape(["Environment reward", "Full-state Nu"], :, 1),
    )
    steps = collect(1:TEST_STEPS)
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
    PlotlyJS.relayout!(plot, Dict{Symbol, Any}(
        :template => "plotly_white",
        :title => "Higher-Ra masked test: $(candidate[:active_inputs]) inputs, q ∈ {$(join(quality_thresholds, ", "))}",
        :width => 1100,
        :height => 500,
        :xaxis => PlotlyJS.attr(title = "Control step"),
        :xaxis2 => PlotlyJS.attr(title = "Control step"),
        :yaxis => PlotlyJS.attr(title = "Mean reward"),
        :yaxis2 => PlotlyJS.attr(title = "Nu"),
    ))
    path = joinpath(directory, "test_curves.svg")
    PlotlyJS.savefig(plot, path; width = 1100, height = 500)
    return path
end

function run_frozen_candidate_test!(options, output, selection_path, frozen, cases)
    index = Int(frozen[:candidate_index])
    candidate = frozen[:candidate]
    checkpoint_path = string(frozen[:checkpoint_path])
    quality_thresholds = Float64.(frozen[:quality_thresholds])
    file_sha256(checkpoint_path) == string(frozen[:checkpoint_sha256]) || error(
        "Frozen candidate checkpoint changed before test: $checkpoint_path",
    )
    checkpoint = JLD2.load(checkpoint_path)
    haskey(checkpoint, "model_payload") || error(
        "Candidate checkpoint has no model_payload: $checkpoint_path",
    )
    candidate_model = checkpoint["model_payload"]
    runtime_flux = Base.invokelatest(() -> getfield(@__MODULE__, :Flux))
    Base.invokelatest(runtime_flux.testmode!, candidate_model)
    input_mask = Float32.(candidate[:mask])
    runtime_env = Base.invokelatest(() -> getfield(@__MODULE__, :env))
    length(input_mask) == size(runtime_env.state, 1) || error(
        "Selected candidate mask has the wrong length.",
    )
    episodes = [
        Base.invokelatest(run_masked_test_episode, candidate_model, input_mask, choice)
        for choice in cases
    ]
    directory = joinpath(output, "test", @sprintf("candidate_%02d", index))
    mkpath(directory)
    csv_path = write_test_csv(joinpath(directory, "test_episodes.csv"), episodes)
    plot_path = make_test_plot(directory, episodes, candidate, quality_thresholds)
    result_path = joinpath(directory, "test_results.jld2")
    config = normalize_archive_dict(JLD2.load(
        joinpath(string(candidate[:source_run_directory]), "config.jld2"),
    )["config"])
    atomic_save(
        result_path;
        schema_version = HR_SCHEMA_VERSION,
        experiment = :higher_ra_masked_apprentice_test,
        study = options.study,
        protocol = study(options.study).protocol,
        rayleigh = study(options.study).rayleigh,
        selection_uses_test_data = false,
        selection_path,
        candidate_index = index,
        quality_thresholds,
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
        checkpoint_sha256 = string(frozen[:checkpoint_sha256]),
        expert_identifier = string(config[:expert_identifier]),
        expert_path = string(config[:expert_path]),
        expert_sha256 = string(config[:expert_sha256]),
        run_file_path = string(config[:run_file_path]),
        run_file_sha256 = string(config[:run_file_sha256]),
        state_corpus_path = string(config[:state_corpus_path]),
        state_corpus_sha256 = string(config[:state_corpus_sha256]),
        expert_baseline_path = string(config[:expert_baseline_path]),
        expert_baseline_sha256 = string(config[:expert_baseline_sha256]),
        unactuated_baseline_path = string(config[:unactuated_baseline_path]),
        unactuated_baseline_sha256 = string(config[:unactuated_baseline_sha256]),
        steps = TEST_STEPS,
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
    return (; index, candidate, quality_thresholds, result_path, csv_path, plot_path)
end

function run_selected_candidate_tests!(options, output, selections)
    frozen = freeze_test_candidates!(output, selections)
    isempty(frozen.candidates) && return (; frozen, tests = NamedTuple[])
    configure_test_runtime!(options, frozen.candidates, output)
    cases = higher_ra_test_cases()
    tests = [
        run_frozen_candidate_test!(options, output, frozen.path, candidate, cases)
        for candidate in frozen.candidates
    ]
    return (; frozen, tests)
end

function analyze_completed_runs(options, jobs)
    output = analysis_directory(
        options.results_root,
        options.study,
        options.experiment_id,
        options.configuration,
    )
    mkpath(output)
    records = reduce(vcat, (load_run_records(options, job) for job in jobs);
                     init = Dict{Symbol, Any}[])
    strengths = observed_strengths(records)
    expected_native_count = length(jobs) *
        length(expected_evaluation_updates(options.expected_updates))
    native_count = count(record -> Symbol(record[:threshold_id]) === :native, records)
    native_count == expected_native_count || error(
        "Expected $expected_native_count native evaluation points, found $native_count.",
    )
    pooled_front = pareto_front(records)
    front_ids = Set(string(record[:candidate_id]) for record in pooled_front)
    csv_path = write_csv(joinpath(output, "evaluations.csv"), records, front_ids)
    front_csv_path = write_csv(
        joinpath(output, "pooled_pareto_front.csv"),
        pooled_front,
        front_ids,
    )
    plot_paths = make_pareto_plot(options, records, pooled_front, output)
    selections = select_quality_candidates(pooled_front)
    test = run_selected_candidate_tests!(options, output, selections)
    selected_ids = [string(entry[:candidate][:candidate_id]) for entry in test.frozen.candidates]
    write_status!(
        analysis_status_path(
            options.results_root,
            options.study,
            options.experiment_id,
            options.configuration,
        );
        state = :complete,
        study = options.study,
        experiment_id = options.experiment_id,
        configuration = options.configuration,
        regularization_strengths = strengths,
        quality_thresholds = collect(HR_QUALITY_THRESHOLDS),
        threshold_selections = test.frozen.threshold_selections,
        unique_test_candidate_count = length(test.tests),
        selected_test_candidates = selected_ids,
        point_count = length(records),
        front_count = length(pooled_front),
        csv_path,
        front_csv_path,
        plot_paths,
        selection_path = test.frozen.path,
        test_result_paths = [entry.result_path for entry in test.tests],
        test_csv_paths = [entry.csv_path for entry in test.tests],
        test_plot_paths = [entry.plot_path for entry in test.tests],
        completed_at = string(Dates.now()),
    )
    println("Completed $(study(options.study).label) analysis for $(options.configuration).")
    println("  points/front: $(length(records)) / $(length(pooled_front))")
    println("  unique selected test candidates: $(length(test.tests))")
    println("  output: $output")
    return (; records, pooled_front, csv_path, front_csv_path, plot_paths,
            selections, test)
end

function analysis_main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    status = analysis_status_path(
        options.results_root,
        options.study,
        options.experiment_id,
        options.configuration,
    )
    try
        write_status!(
            status;
            state = :waiting,
            study = options.study,
            experiment_id = options.experiment_id,
            configuration = options.configuration,
            regularization_strengths = options.strengths,
            quality_thresholds = collect(HR_QUALITY_THRESHOLDS),
            started_at = string(Dates.now()),
        )
        jobs = wait_for_runs(options)
        return analyze_completed_runs(options, jobs)
    catch exception
        write_status!(
            status;
            state = :failed,
            study = options.study,
            experiment_id = options.experiment_id,
            configuration = options.configuration,
            regularization_strengths = options.strengths,
            error = sprint(showerror, exception),
            failed_at = string(Dates.now()),
        )
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    analysis_main()
end
