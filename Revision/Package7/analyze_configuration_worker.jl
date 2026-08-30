using Dates
using JLD2
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "Package7Study.jl"))
using .Package7Study

const P7_DISTILLATION_DIRECTORY = joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation")
include(joinpath(P7_DISTILLATION_DIRECTORY, "ParetoArchive.jl"))

const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "results")
const P7_TEST_STEPS = 200
const THRESHOLD_COLOR_PALETTE = ("#2166AC", "#92C5DE", "#D6604D", "#67001F")
const REPLICATE_SYMBOLS = Dict(1 => "circle", 2 => "diamond", 3 => "square")

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "experiment_id" => nothing,
        "config" => nothing,
        "strengths" => Float64[],
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "poll_seconds" => 60,
        "timeout_seconds" => 14 * 24 * 60 * 60,
        "expected_updates" => P7_UPDATES,
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
        for strength in options.strengths for replicate in P7_REPLICATES
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
            error("Package-7 training failed:\n$details")
        end
        all_complete = all((!isnothing(status) && Symbol(status[:state]) === :complete) for (_, status) in statuses)
        all_complete && return jobs
        time() >= deadline && error("Timed out waiting for Package-7 runs $(join((job.id for job in jobs), ", ")).")
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
        Symbol(config[:experiment]) === :package7_fixed_regularizer_comparison,
        string(config[:experiment_id]) == options.experiment_id,
        string(config[:configuration]) == job.configuration,
        Symbol(config[:method]) === job.method,
        Bool(config[:group_channels]) == job.group_channels,
        Int(config[:replicate]) == job.replicate,
        Float64(config[:regularization_strength]) == job.regularization_strength,
        Int(config[:master_seed]) == P7_MASTER_SEED,
        Int(config[:apprentice_seed]) == job.apprentice_seed,
        Int(config[:batch_order_seed]) == job.batch_seed,
        Int(config[:regularized_updates]) == options.expected_updates,
        Symbol(config[:threshold_importance_mode]) === :max_input_l1,
        Int(config[:threshold_minimum_active_groups]) == 1,
    )
    all(checks) || error("Run $(job.id) does not match its requested Package-7 configuration.")
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

function write_csv(path::AbstractString, records, front_ids)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "run_id,replicate,configuration,strength,update,candidate_id,threshold_id,threshold_value,active_groups,active_inputs,validation_matching,pooled_pareto,under_quality_threshold")
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
                string(Float64(record[:validation_matching]) <= P7_QUALITY_THRESHOLD),
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

function make_plot(options, records, pooled_front, output_directory)
    ensure_plotly_loaded!()
    return Base.invokelatest(make_plot_loaded, options, records, pooled_front, output_directory)
end

function make_plot_loaded(options, records, pooled_front, output_directory)
    strengths = observed_strengths(records)
    thresholds = observed_thresholds(records)
    colors = threshold_colors(thresholds)
    traces = PlotlyJS.GenericTrace[]
    shown_thresholds = Set{Float64}()
    for replicate in P7_REPLICATES, threshold in thresholds
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
    for replicate in P7_REPLICATES
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
        title = "Package 7 $(options.configuration), λ ∈ {$(join(strengths, ", "))}",
        xaxis = PlotlyJS.attr(title = "Active groups"),
        yaxis = PlotlyJS.attr(title = "Validation expert-action matching (MSE)", type = "log"),
        legend = PlotlyJS.attr(title = PlotlyJS.attr(text = "Threshold")),
        shapes = [PlotlyJS.attr(
            type = "line",
            xref = "paper",
            x0 = 0,
            x1 = 1,
            yref = "y",
            y0 = P7_QUALITY_THRESHOLD,
            y1 = P7_QUALITY_THRESHOLD,
            line = PlotlyJS.attr(color = "#555555", width = 1.5, dash = "dash"),
        )],
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

function select_sparse_test_candidate(pooled_front)
    qualified = filter(
        record -> Float64(record[:validation_matching]) <= P7_QUALITY_THRESHOLD,
        pooled_front,
    )
    isempty(qualified) && return nothing
    return first(sort(qualified; by = record -> (
        Int(record[:active_inputs]),
        Int(record[:active_groups]),
        Float64(record[:validation_matching]),
        Int(record[:update]),
        string(record[:run_id]),
        string(record[:candidate_id]),
    )))
end

function freeze_test_candidate!(output, candidate, checkpoint_path)
    path = joinpath(output, "selected_test_candidate.jld2")
    atomic_save(
        path;
        schema_version = P7_SCHEMA_VERSION,
        experiment = :package7_fixed_regularizer_comparison,
        selection_source = :pooled_validation_pareto_front,
        selection_rule = :minimum_active_inputs_under_quality_threshold,
        quality_threshold = P7_QUALITY_THRESHOLD,
        selection_uses_test_data = false,
        candidate,
        checkpoint_path,
        checkpoint_sha256 = file_sha256(checkpoint_path),
        frozen_before_test = true,
        frozen_at = string(Dates.now()),
    )
    return path
end

function configure_test_runtime!(candidate, output)
    run_directory = string(candidate[:source_run_directory])
    config = normalize_archive_dict(JLD2.load(joinpath(run_directory, "config.jld2"))["config"])
    expert_path = string(config[:expert_path])
    ENV["DISTILLATION_PROTOCOL"] = "fixed"
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(Bool(config[:group_channels]))
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV["DISTILLATION_FIXED_EXPERT_PATH"] = expert_path
    ENV["REVISION_RUN_SEED"] = string(config[:apprentice_seed])
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output, "test", "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output, "test", "apprentice_output")
    Base.include(@__MODULE__, joinpath(P7_DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
    expert_metadata = Base.invokelatest(() -> getfield(@__MODULE__, :DISTILLATION_EXPERT_METADATA))
    string(expert_metadata[:identifier]) == string(config[:expert_identifier]) || error(
        "Loaded Fixed expert does not match the selected candidate's training corpus.",
    )
    return config
end


function normalize_test_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 && (values = dropdims(values; dims = 3))
    length(values) == 12 || error("Expected twelve actuator actions, got $(size(values)).")
    return vec(values)
end

function run_masked_test_episode(candidate_model, input_mask)
    runtime_rl = Base.invokelatest(() -> getfield(@__MODULE__, :RL))
    runtime_env = Base.invokelatest(() -> getfield(@__MODULE__, :env))
    initialize_episode = Base.invokelatest(() -> getfield(@__MODULE__, :generate_random_init))
    nusselt_function = Base.invokelatest(() -> getfield(@__MODULE__, :state_Nu))
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    Base.invokelatest(initialize_episode)
    rewards = Vector{Float64}(undef, P7_TEST_STEPS)
    state_nusselt = Vector{Float64}(undef, P7_TEST_STEPS)
    actions = Matrix{Float32}(undef, P7_TEST_STEPS, 12)
    for step in 1:P7_TEST_STEPS
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
        isfinite(rewards[step]) && isfinite(state_nusselt[step]) || error(
            "Non-finite Package-7 test value at step $step.",
        )
    end
    return (; rewards, state_nusselt, actions)
end

function write_test_csv(path, episode)
    open(path, "w") do io
        println(io, "step,reward,state_nusselt")
        for step in 1:P7_TEST_STEPS
            println(io, "$(step),$(episode.rewards[step]),$(episode.state_nusselt[step])")
        end
    end
    return path
end

function make_test_plot(output, episode, candidate)
    ensure_plotly_loaded!()
    return Base.invokelatest(make_test_plot_loaded, output, episode, candidate)
end

function make_test_plot_loaded(output, episode, candidate)
    plot = PlotlyJS.make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = reshape(["Environment reward", "Full-state Nu"], :, 1),
    )
    steps = collect(1:P7_TEST_STEPS)
    PlotlyJS.add_trace!(plot, PlotlyJS.scatter(
        x = steps, y = episode.rewards, mode = "lines", name = "Reward",
        line = PlotlyJS.attr(color = "#277DA1", width = 2.5),
    ); row = 1, col = 1)
    PlotlyJS.add_trace!(plot, PlotlyJS.scatter(
        x = steps, y = episode.state_nusselt, mode = "lines", name = "state_Nu",
        line = PlotlyJS.attr(color = "#F2A13A", width = 2.5),
    ); row = 1, col = 2)
    PlotlyJS.relayout!(plot, Dict{Symbol, Any}(
        :template => "plotly_white",
        :title => "Masked P7 test: $(candidate[:active_inputs]) active inputs",
        :width => 1100,
        :height => 500,
        :xaxis => PlotlyJS.attr(title = "Control step"),
        :xaxis2 => PlotlyJS.attr(title = "Control step"),
        :yaxis => PlotlyJS.attr(title = "Mean reward"),
        :yaxis2 => PlotlyJS.attr(title = "Nu"),
    ))
    path = joinpath(output, "test", "test_curves.svg")
    PlotlyJS.savefig(plot, path; width = 1100, height = 500)
    return path
end

function run_selected_candidate_test!(output, selected)
    run_directory = string(selected[:source_run_directory])
    checkpoint_path = candidate_checkpoint_for_record(run_directory, selected)
    candidate = hydrate_candidate_record(selected, run_directory)
    selection_path = freeze_test_candidate!(output, candidate, checkpoint_path)
    test_directory = joinpath(output, "test")
    mkpath(test_directory)
    config = configure_test_runtime!(candidate, output)
    checkpoint = JLD2.load(checkpoint_path)
    haskey(checkpoint, "model_payload") || error("Candidate checkpoint has no model_payload: $checkpoint_path")
    candidate_model = checkpoint["model_payload"]
    runtime_flux = Base.invokelatest(() -> getfield(@__MODULE__, :Flux))
    Base.invokelatest(runtime_flux.testmode!, candidate_model)
    input_mask = Float32.(candidate[:mask])
    runtime_env = Base.invokelatest(() -> getfield(@__MODULE__, :env))
    length(input_mask) == size(runtime_env.state, 1) || error("Selected candidate mask has the wrong length.")
    episode = Base.invokelatest(run_masked_test_episode, candidate_model, input_mask)
    csv_path = write_test_csv(joinpath(test_directory, "test_episode.csv"), episode)
    plot_path = make_test_plot(output, episode, candidate)
    result_path = joinpath(test_directory, "test_results.jld2")
    atomic_save(
        result_path;
        schema_version = P7_SCHEMA_VERSION,
        experiment = :package7_masked_apprentice_test,
        protocol = :fixed,
        selection_uses_test_data = false,
        selection_path,
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
        steps = P7_TEST_STEPS,
        rewards = episode.rewards,
        state_nusselt = episode.state_nusselt,
        actions = episode.actions,
        reward_sum = sum(episode.rewards),
        mean_reward = mean(episode.rewards),
        sum_state_nusselt = sum(episode.state_nusselt),
        negative_sum_state_nusselt = -sum(episode.state_nusselt),
        mean_state_nusselt = mean(episode.state_nusselt),
        csv_path,
        plot_path,
        completed_at = string(Dates.now()),
    )
    return (; candidate, selection_path, result_path, csv_path, plot_path)
end

function clear_selected_candidate_test!(output)
    for path in (
        joinpath(output, "selected_test_candidate.jld2"),
        joinpath(output, "test", "test_results.jld2"),
        joinpath(output, "test", "test_episode.csv"),
        joinpath(output, "test", "test_curves.svg"),
    )
        isfile(path) && rm(path; force = true)
    end
    return nothing
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
    plot_paths = make_plot(options, records, pooled_front, output)
    selected = select_sparse_test_candidate(pooled_front)
    test = if isnothing(selected)
        clear_selected_candidate_test!(output)
        nothing
    else
        run_selected_candidate_test!(output, selected)
    end
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
        selected_test_candidate = isnothing(test) ? nothing : string(test.candidate[:candidate_id]),
        selected_test_active_inputs = isnothing(test) ? nothing : Int(test.candidate[:active_inputs]),
        selected_test_validation_matching = isnothing(test) ? nothing : Float64(test.candidate[:validation_matching]),
        selection_path = isnothing(test) ? nothing : test.selection_path,
        test_result_path = isnothing(test) ? nothing : test.result_path,
        test_csv_path = isnothing(test) ? nothing : test.csv_path,
        test_plot_path = isnothing(test) ? nothing : test.plot_path,
        completed_at = string(Dates.now()),
    )
    println("Completed Package-7 Pareto analysis for $(options.configuration), λ ∈ {$(join(strengths, ", "))}.")
    println("  points/front: $(length(records)) / $(length(pooled_front))")
    isnothing(test) && println("  selected test candidate: NR (no point under quality threshold)")
    println("  output: $output")
    return (; records, pooled_front, csv_path, front_csv_path, plot_paths, test)
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
