using Dates
using JLD2
using Printf

include(joinpath(@__DIR__, "Package7Study.jl"))
using .Package7Study

const P7_DISTILLATION_DIRECTORY = joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation")
include(joinpath(P7_DISTILLATION_DIRECTORY, "ParetoArchive.jl"))

const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "results")
const THRESHOLD_COLORS = Dict(
    0.0 => "#2166AC",
    0.0015 => "#92C5DE",
    0.003 => "#D6604D",
    0.005 => "#67001F",
)
const REPLICATE_SYMBOLS = Dict(1 => "circle", 2 => "diamond", 3 => "square")

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "config" => nothing,
        "strength" => nothing,
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
            println("Usage: analyze_configuration_worker.jl --config NAME --strength VALUE [--results-dir PATH] [--poll-seconds N] [--timeout-seconds N]")
            return nothing
        elseif argument == "--retry-failed"
            values["retry_failed"] = true
            index += 1
            continue
        end
        index == length(arguments) && error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    isnothing(values["config"]) && error("--config is required.")
    isnothing(values["strength"]) && error("--strength is required.")
    return (
        configuration = normalize_configuration(values["config"]),
        strength = parse(Float64, string(values["strength"])),
        results_root = abspath(string(values["results_dir"])),
        poll_seconds = parse(Int, string(values["poll_seconds"])),
        timeout_seconds = parse(Int, string(values["timeout_seconds"])),
        expected_updates = parse(Int, string(values["expected_updates"])),
        retry_failed = Bool(values["retry_failed"]),
    )
end

function wait_for_runs(options)
    jobs = [job_for(options.configuration, options.strength, replicate; updates = options.expected_updates) for replicate in P7_REPLICATES]
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
        string(config[:configuration]) == job.configuration,
        Symbol(config[:method]) === job.method,
        Bool(config[:group_channels]) == job.group_channels,
        Int(config[:replicate]) == job.replicate,
        Float64(config[:regularization_strength]) == job.regularization_strength,
        Int(config[:master_seed]) == P7_MASTER_SEED,
        Int(config[:apprentice_seed]) == job.apprentice_seed,
        Int(config[:batch_order_seed]) == job.batch_seed,
        Int(config[:regularized_updates]) == options.expected_updates,
        Float64.(config[:threshold_values]) == collect(P7_THRESHOLDS),
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
        record[:regularization_strength] = job.regularization_strength
        push!(records, record)
    end
    return retain_successful_threshold_records(records; context = job.id)
end

function write_csv(path::AbstractString, records, front_ids)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "run_id,replicate,configuration,strength,update,candidate_id,threshold_id,threshold_value,active_groups,active_inputs,validation_matching,pooled_pareto")
        for record in records
            @printf(
                io,
                "%s,%d,%s,%.12g,%d,%s,%s,%.12g,%d,%d,%.17g,%s\n",
                string(record[:run_id]), Int(record[:replicate]), string(record[:configuration]),
                Float64(record[:regularization_strength]), Int(record[:update]),
                string(record[:candidate_id]), string(record[:threshold_id]),
                Float64(record[:threshold_value]), Int(record[:active_groups]),
                Int(record[:active_inputs]), Float64(record[:validation_matching]),
                string(string(record[:candidate_id]) in front_ids),
            )
        end
    end
    return path
end

function ensure_plotly_loaded!()
    isdefined(@__MODULE__, :PlotlyJS) || Base.eval(@__MODULE__, :(using PlotlyJS))
    return nothing
end

function make_plot(options, records, pooled_front, output_directory)
    ensure_plotly_loaded!()
    return Base.invokelatest(make_plot_loaded, options, records, pooled_front, output_directory)
end

function make_plot_loaded(options, records, pooled_front, output_directory)
    traces = PlotlyJS.GenericTrace[]
    for replicate in P7_REPLICATES, threshold in P7_THRESHOLDS
        selected = filter(record -> Int(record[:replicate]) == replicate && Float64(record[:threshold_value]) == threshold, records)
        active_groups = Int.(getindex.(selected, :active_groups))
        active_inputs = Int.(getindex.(selected, :active_inputs))
        push!(traces, PlotlyJS.scatter(
            x = active_groups,
            y = Float64.(getindex.(selected, :validation_matching)),
            mode = "markers",
            name = "τ=$(threshold)",
            legendgroup = "threshold_$(threshold)",
            showlegend = replicate == 1,
            marker = PlotlyJS.attr(
                color = THRESHOLD_COLORS[threshold],
                symbol = REPLICATE_SYMBOLS[replicate],
                size = 5,
                opacity = 0.38,
                line = PlotlyJS.attr(width = 0),
            ),
            customdata = hcat(active_groups, active_inputs, Int.(getindex.(selected, :update)), fill(replicate, length(selected))),
            hovertemplate = "groups=%{customdata[0]}<br>global inputs=%{customdata[1]}<br>MSE=%{y:.4e}<br>update=%{customdata[2]}<br>replicate=%{customdata[3]}<extra></extra>",
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
        title = "Package 7 $(options.configuration), λ=$(options.strength)",
        xaxis = PlotlyJS.attr(title = "Active groups"),
        yaxis = PlotlyJS.attr(title = "Validation expert-action matching (MSE)", type = "log"),
        legend = PlotlyJS.attr(title = PlotlyJS.attr(text = "Threshold")),
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

function analyze_completed_runs(options, jobs)
    output = analysis_directory(options.results_root, options.configuration, options.strength)
    mkpath(output)
    records = reduce(vcat, (load_run_records(options, job) for job in jobs); init = Dict{Symbol, Any}[])
    expected_native_count = length(P7_REPLICATES) * length(expected_evaluation_updates(options.expected_updates))
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
    data_path = atomic_save(
        joinpath(output, "pareto_points.jld2");
        schema_version = P7_SCHEMA_VERSION,
        experiment = :package7_fixed_regularizer_comparison,
        configuration = options.configuration,
        regularization_strength = options.strength,
        threshold_values = collect(P7_THRESHOLDS),
        threshold_colors = copy(THRESHOLD_COLORS),
        records,
        pooled_front,
        generated_at = string(Dates.now()),
    )
    plot_paths = make_plot(options, records, pooled_front, output)
    write_status!(
        analysis_status_path(options.results_root, options.configuration, options.strength);
        state = :complete,
        configuration = options.configuration,
        regularization_strength = options.strength,
        point_count = length(records),
        front_count = length(pooled_front),
        csv_path,
        front_csv_path,
        data_path,
        plot_paths,
        completed_at = string(Dates.now()),
    )
    println("Completed Package-7 Pareto analysis for $(options.configuration), λ=$(options.strength).")
    println("  points/front: $(length(records)) / $(length(pooled_front))")
    println("  output: $output")
    return (; records, pooled_front, csv_path, front_csv_path, data_path, plot_paths)
end

function analysis_main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    status = analysis_status_path(options.results_root, options.configuration, options.strength)
    try
        write_status!(status; state = :waiting, configuration = options.configuration,
                      regularization_strength = options.strength, started_at = string(Dates.now()))
        jobs = wait_for_runs(options)
        return analyze_completed_runs(options, jobs)
    catch exception
        write_status!(status; state = :failed, configuration = options.configuration,
                      regularization_strength = options.strength,
                      error = sprint(showerror, exception), failed_at = string(Dates.now()))
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    analysis_main()
end
