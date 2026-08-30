using Dates
using JLD2
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "Package6Study.jl"))
include(joinpath(@__DIR__, "Package6Analysis.jl"))
if !isdefined(@__MODULE__, :PARETO_ARCHIVE_SCHEMA_VERSION)
    include(joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation", "ParetoArchive.jl"))
end
using .Package6Study
using .Package6Analysis

const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "results", "study")
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const HIGHLIGHT_TARGETS = (48, 24, 12, 6, 3, 1)
const COLORS = ("#2166AC", "#4393C3", "#92C5DE", "#D6604D", "#B2182B")

function ensure_plotly_loaded!()
    isdefined(@__MODULE__, :PlotlyJS) || Base.eval(@__MODULE__, :(using PlotlyJS))
    return nothing
end

function analysis_usage(io::IO = stdout)
    println(io, """
    Usage: julia --project=. analyze_study_worker.jl --protocol fixed|varying
           [--results-dir PATH] [--poll-seconds N] [--timeout-seconds N]
           [--skip-test] [--parallel-test]

    The production launcher uses a 60-second poll and a 14-day timeout.
    --skip-test is intended only for offline replotting and automated tests.
    """)
end

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "protocol" => nothing,
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "poll_seconds" => P6_POLL_SECONDS,
        "timeout_seconds" => P6_TIMEOUT_SECONDS,
        "skip_test" => false,
        "parallel_test" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            analysis_usage()
            return nothing
        elseif argument == "--skip-test"
            values["skip_test"] = true
            index += 1
        elseif argument == "--parallel-test"
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
    return (
        protocol = normalize_protocol(values["protocol"]),
        results_root = abspath(string(values["results_dir"])),
        poll_seconds = parse(Int, string(values["poll_seconds"])),
        timeout_seconds = parse(Int, string(values["timeout_seconds"])),
        skip_test = Bool(values["skip_test"]),
        parallel_test = Bool(values["parallel_test"]),
    )
end

analysis_directory(options) = joinpath(options.results_root, string(options.protocol), "analysis")

function write_failure_report(options, failed, missing, running)
    path = joinpath(analysis_directory(options), "failure_report.md")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "# Package 6 $(options.protocol) analysis failure\n")
        println(io, "Generated: $(Dates.now())\n")
        if !isempty(failed)
            println(io, "## Explicitly failed training runs\n")
            for (job, status) in failed
                println(io, "- `$(job.id)`: $(get(status, :error_message, "unknown failure"))")
            end
            println(io)
        end
        !isempty(missing) && println(io, "## Missing statuses\n\n", join("- `$(job.id)`\n" for job in missing))
        !isempty(running) && println(io, "## Running statuses\n\n", join("- `$(job.id)`\n" for job in running))
        println(io, "Repair or explicitly restart the affected worker, then restart this analysis worker.")
    end
    return path
end

function wait_for_runs(options)
    jobs = study_jobs(options.protocol)
    length(jobs) == 18 || error("Expected exactly 18 training jobs for $(options.protocol).")
    started = time()
    while true
        failed = Tuple[]
        missing = NamedTuple[]
        running = NamedTuple[]
        complete = NamedTuple[]
        for job in jobs
            status = load_status(status_path(options.results_root, job))
            if isnothing(status)
                push!(missing, job)
            elseif status[:state] === :failed
                push!(failed, (job, status))
            elseif status[:state] === :complete
                push!(complete, job)
            else
                push!(running, job)
            end
        end
        if !isempty(failed)
            report = write_failure_report(options, failed, missing, running)
            error("$(length(failed)) training run(s) failed explicitly. See $report")
        end
        length(complete) == length(jobs) && return jobs
        if time() - started >= options.timeout_seconds
            report = write_failure_report(options, failed, missing, running)
            error("Timed out waiting for training runs. See $report")
        end
        println("$(Dates.now()): complete=$(length(complete))/18, running=$(length(running)), missing=$(length(missing)); polling again in $(options.poll_seconds)s")
        sleep(options.poll_seconds)
    end
end

function load_run(options, job)
    directory = run_directory(options.results_root, job)
    status = load_status(status_path(options.results_root, job))
    status[:state] === :complete || error("Run $(job.id) is not complete.")
    config_loaded = JLD2.load(joinpath(directory, "config.jld2"))
    config = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in config_loaded["config"])
    config_fingerprint = string(config_loaded["config_fingerprint"])
    string(status[:config_fingerprint]) == config_fingerprint || error("Status/config fingerprint mismatch for $(job.id).")
    summary = JLD2.load(joinpath(directory, "summary.jld2"))
    string(summary["config_fingerprint"]) == config_fingerprint || error("Summary/config fingerprint mismatch for $(job.id).")
    collection = load_evaluation_collection(directory)
    collection.run_id == job.id || error("Evaluation run ID mismatch for $(job.id).")
    collection.config_fingerprint == config_fingerprint || error("Evaluation fingerprint mismatch for $(job.id).")
    updates = Int[]
    records = Dict{Symbol, Any}[]
    for batch in collection.batches
        update = Int(batch[:update])
        push!(updates, update)
        candidates = batch[:candidates]
        length(candidates) == 1 || error("Native-only run $(job.id) has $(length(candidates)) candidates at update $update.")
        record = normalize_record(only(candidates); metadata = (
            run_id = job.id,
            protocol = job.protocol,
            method = job.method,
            strength_index = job.strength_index,
            regularization_strength = job.regularization_strength,
            replicate = job.replicate,
            source_run_directory = directory,
        ))
        Symbol(record[:threshold_id]) === :native || error("Non-native candidate in $(job.id).")
        push!(records, record)
    end
    expected = expected_evaluation_updates(job.updates)
    updates == expected || error("Evaluation coverage mismatch for $(job.id): expected $(length(expected)), found $(length(updates)).")
    return (; job, directory, status, config, config_fingerprint, summary, records)
end

function audit_runs(options, jobs)
    runs = [load_run(options, job) for job in jobs]
    count(run -> run.job.method === :go, runs) == 15 || error("Audit expected 15 GO runs.")
    count(run -> run.job.method === :gr, runs) == 3 || error("Audit expected 3 GR runs.")
    expected_keys = Dict(job.id => job for job in study_jobs(options.protocol))
    Set(run.job.id for run in runs) == Set(keys(expected_keys)) || error("Run identity set mismatch.")
    for run in runs
        config = run.config
        Symbol(config[:scientific_scope]) === :package6 || error("Smoke/non-production config in $(run.job.id).")
        Symbol(config[:protocol]) === options.protocol || error("Protocol mismatch in $(run.job.id).")
        Symbol(config[:method]) === run.job.method || error("Method mismatch in $(run.job.id).")
        Bool(config[:group_channels]) == false || error("Package 6 must use separate channels.")
        Bool(config[:native_sparsity_only]) || error("Package 6 must use native sparsity.")
        Int(config[:regularized_updates]) == run.job.updates || error("Update budget mismatch in $(run.job.id).")
        Int(config[:batch_size]) == P6_TRAINING_BATCH_SIZE[options.protocol] || error("Training batch-size mismatch in $(run.job.id).")
        Int(config[:validation_batch_size]) == P6_VALIDATION_BATCH_SIZE[options.protocol] || error("Validation batch-size mismatch in $(run.job.id).")
        Int(config[:evaluation_interval]) == P6_EVALUATION_INTERVAL || error("Evaluation interval mismatch.")
        Float64(config[:regression_learning_rate]) == P6_REGRESSION_LEARNING_RATE || error("Regression LR mismatch.")
        Float64.(config[:go_strength_grid]) == collect(P6_STRENGTHS[options.protocol]) || error("GO strength-grid mismatch.")
        # This threshold controls analysis-time candidate acceptance only. Its
        # archived launch value must not prevent reanalysis with a new limit.
        haskey(config, :quality_threshold) || error("Stored quality-threshold metadata is missing.")
        Int(config[:apprentice_seed]) == run.job.apprentice_seed || error("Apprentice seed mismatch.")
        Int(config[:batch_order_seed]) == run.job.batch_seed || error("Batch seed mismatch.")
        string(config[:pairing_hash]) == run.job.pairing_hash || error("Pairing hash mismatch.")
        Int(run.status[:update]) == run.job.updates || error("Incomplete final update status in $(run.job.id).")
    end
    expert_ids = unique(string(run.config[:expert_identifier]) for run in runs)
    length(expert_ids) == 1 || error("Runs use different experts.")
    for split in (:train_source_files, :validation_source_files, :test_source_files)
        identities = unique(fingerprint(run.config[split]) for run in runs)
        length(identities) == 1 || error("Runs use different $split corpora.")
    end
    for replicate in 1:3
        paired = filter(run -> run.job.replicate == replicate, runs)
        length(paired) == 6 || error("Replicate $replicate does not contain five GO and one GR run.")
        length(unique(Int(run.config[:apprentice_seed]) for run in paired)) == 1 || error("Apprentice pairing failed for replicate $replicate.")
        length(unique(Int(run.config[:batch_order_seed]) for run in paired)) == 1 || error("Batch pairing failed for replicate $replicate.")
        hashes = unique(string(run.config[:initial_apprentice_parameter_hash]) for run in paired)
        length(hashes) == 1 || error("Initial apprentice parameter hashes differ within replicate $replicate.")
    end
    length(unique(Int(run.config[:apprentice_seed]) for run in runs)) == 3 || error("Replicate apprentice seeds are not distinct.")
    length(unique(Int(run.config[:batch_order_seed]) for run in runs)) == 3 || error("Replicate batch seeds are not distinct.")
    return (runs, expert_identifier = only(expert_ids), audit_passed = true)
end

function csv_escape(value)
    value === missing && return ""
    text = string(value)
    occursin(r"[\",\n\r]", text) || return text
    return "\"" * replace(text, "\"" => "\"\"") * "\""
end

function write_csv(path, rows)
    rows = collect(rows)
    mkpath(dirname(path))
    keys_order = isempty(rows) ? Symbol[] : sort!(unique(vcat([collect(keys(row)) for row in rows]...)); by = string)
    open(path, "w") do io
        println(io, join(string.(keys_order), ','))
        for row in rows
            println(io, join((csv_escape(haskey(row, key) ? row[key] : missing) for key in keys_order), ','))
        end
    end
    return path
end

function build_metrics(audit)
    runs = audit.runs
    protocol = only(unique(run.job.protocol for run in runs))
    quality_threshold = P6_QUALITY_THRESHOLDS[protocol]
    run_fronts = Dict(run.job.id => scientific_front(run.records) for run in runs)
    strength_fronts = Dict{Tuple{Symbol, Int}, Vector{Dict{Symbol, Any}}}()
    for method in (:go, :gr)
        indices = method === :go ? (1:5) : (0:0)
        for strength_index in indices
            records = reduce(vcat, [run.records for run in runs if run.job.method === method && run.job.strength_index == strength_index])
            strength_fronts[(method, strength_index)] = scientific_front(records)
        end
    end
    global_fronts = Dict(method => scientific_front(reduce(vcat, [run.records for run in runs if run.job.method === method])) for method in (:go, :gr))
    checkpoints = Dict{String, Any}()
    late_rows = NamedTuple[]
    hitting_rows = NamedTuple[]
    reset_rows = NamedTuple[]
    reset_event_rows = NamedTuple[]
    archive_rows = NamedTuple[]
    run_summary_rows = NamedTuple[]
    for run in runs
        own_front = run_fronts[run.job.id]
        strength_front = strength_fronts[(run.job.method, run.job.strength_index)]
        measured = checkpoint_metrics(run.records, own_front, strength_front)
        checkpoints[run.job.id] = measured
        for row in late_metrics(measured, run.job.updates)
            push!(late_rows, merge((run_id = run.job.id, method = run.job.method, strength_index = run.job.strength_index, strength = run.job.regularization_strength, replicate = run.job.replicate), row))
        end
        for row in hitting_metrics(run.records)
            push!(hitting_rows, merge((run_id = run.job.id, method = run.job.method, strength_index = run.job.strength_index, strength = run.job.regularization_strength, replicate = run.job.replicate), row))
        end
        resets = reset_metrics(run.records, run.job.updates)
        push!(reset_rows, merge((run_id = run.job.id, method = run.job.method, strength_index = run.job.strength_index, strength = run.job.regularization_strength, replicate = run.job.replicate), resets.summary))
        append!(reset_event_rows, [merge((run_id = run.job.id, method = run.job.method, strength_index = run.job.strength_index, replicate = run.job.replicate), event) for event in resets.events])
        convergence = archive_convergence(run.records, own_front)
        append!(archive_rows, [merge((run_id = run.job.id, method = run.job.method, strength_index = run.job.strength_index, strength = run.job.regularization_strength, replicate = run.job.replicate, updates_to_90 = convergence.updates_to_90, updates_to_100 = convergence.updates_to_100), row) for row in convergence.rows])
        valid = filter(record -> record[:numeric_status] === :ok && isfinite(record[:validation_matching]), run.records)
        best = first(sort(valid; by = record -> (record[:validation_matching], record[:update])))
        sparsest = first(sort(valid; by = record -> (record[:active_groups], record[:validation_matching], record[:update])))
        late20 = only(filter(row -> row.run_id == run.job.id && row.window_fraction == 0.20, late_rows))
        push!(run_summary_rows, (
            run_id = run.job.id, method = run.job.method, strength_index = run.job.strength_index,
            strength = run.job.regularization_strength, replicate = run.job.replicate,
            runtime_seconds = Float64(run.summary["elapsed_seconds"]), evaluation_count = length(run.records),
            numeric_failure_count = count(record -> record[:numeric_status] !== :ok || !isfinite(record[:validation_matching]), run.records),
            front_size = length(own_front), best_validation_mse = best[:validation_matching], best_mse_groups = best[:active_groups],
            sparsest_groups = sparsest[:active_groups], sparsest_validation_mse = sparsest[:validation_matching],
            late20_front_near_fraction = late20.front_near_fraction,
            late20_median_strength_regret = late20.median_strength_front_regret,
        ))
    end
    attainment_rows = NamedTuple[]
    for ((method, strength_index), _) in strength_fronts
        matching_runs = filter(run -> run.job.method === method && run.job.strength_index == strength_index, runs)
        fronts = Dict(run.job.replicate => run_fronts[run.job.id] for run in matching_runs)
        strength = only(unique(run.job.regularization_strength for run in matching_runs))
        append!(attainment_rows, [merge((; method, strength_index, strength), row) for row in empirical_attainment(fronts)])
    end
    global_attainment_rows = NamedTuple[]
    for method in (:go, :gr)
        fronts = Dict(replicate => scientific_front(reduce(vcat, [run.records for run in runs if run.job.method === method && run.job.replicate == replicate])) for replicate in 1:3)
        append!(global_attainment_rows, [merge((; method), row) for row in empirical_attainment(fronts)])
    end
    trend_rows = NamedTuple[]
    for replicate in 1:3
        summaries = sort(filter(row -> row.method === :go && row.replicate == replicate, run_summary_rows); by = row -> row.strength)
        strengths = [row.strength for row in summaries]
        for metric in (:best_validation_mse, :sparsest_groups, :late20_front_near_fraction, :late20_median_strength_regret)
            values = [getproperty(row, metric) for row in summaries]
            push!(trend_rows, (replicate, metric, spearman_rho = spearman_correlation(strengths, values), strengths = join(strengths, ';'), values = join(values, ';')))
        end
    end
    masks = Dict{Symbol, Any}()
    hydrated = Dict{String, Dict{Symbol, Any}}()
    hydrate(record) = get!(hydrated, string(record[:candidate_id])) do
        hydrate_candidate_record(
            record,
            string(record[:source_run_directory]);
            required_keys = (:global_mask,),
        )
    end
    for method in (:go, :gr)
        selected_runs = filter(run -> run.job.method === method, runs)
        masks[method] = mask_stability(
            Dict(run.job.id => run_fronts[run.job.id] for run in selected_runs),
            Dict(run.job.id => run.records for run in selected_runs),
            mse_threshold = quality_threshold,
            hydrate = hydrate,
        )
    end
    hitting_summary_rows = NamedTuple[]
    for method in (:go, :gr), strength_index in (method === :go ? (1:5) : (0:0)), target in 0:96
        rows = filter(row -> row.method === method && row.strength_index == strength_index && row.target_groups == target, hitting_rows)
        isempty(rows) && continue
        reachable_rows = filter(row -> row.reachable, rows)
        push!(hitting_summary_rows, (
            method,
            strength_index,
            strength = first(rows).strength,
            target_groups = target,
            reachability_rate = mean(row.reachable for row in rows),
            median_first_update = isempty(reachable_rows) ? missing : median(Float64(row.first_update) for row in reachable_rows),
            median_validation_mse_at_hit = isempty(reachable_rows) ? missing : median(Float64(row.validation_mse_at_hit) for row in reachable_rows),
        ))
    end
    return (; run_fronts, strength_fronts, global_fronts, checkpoints, late_rows, hitting_rows, hitting_summary_rows, reset_rows, reset_event_rows, archive_rows, run_summary_rows, attainment_rows, global_attainment_rows, trend_rows, masks)
end

function strength_color(index)
    index == 0 && return "#444444"
    return COLORS[index]
end

function save_svg(plot, path; width = 950, height = 580)
    mkpath(dirname(path))
    PlotlyJS.savefig(plot, path; width, height)
    return path
end

function make_plots(options, audit, metrics)
    ensure_plotly_loaded!()
    return Base.invokelatest(make_plots_loaded, options, audit, metrics)
end

function make_plots_loaded(options, audit, metrics)
    output = joinpath(analysis_directory(options), "plots")
    mkpath(output)
    paths = Dict{Symbol, Any}()
    strengths = P6_STRENGTHS[options.protocol]
    quality_threshold = P6_QUALITY_THRESHOLDS[options.protocol]
    pareto_traces = PlotlyJS.GenericTrace[]
    for run in audit.runs
        front = metrics.run_fronts[run.job.id]
        push!(pareto_traces, scatter(
            x = [record[:active_groups] for record in front], y = [record[:validation_matching] for record in front],
            mode = "lines+markers", name = run.job.id, legendgroup = string(run.job.method, run.job.strength_index),
            showlegend = run.job.replicate == 1, line = attr(color = strength_color(run.job.strength_index), width = 1),
            marker = attr(size = 4), opacity = 0.35,
        ))
    end
    for method in (:go, :gr)
        front = metrics.global_fronts[method]
        push!(pareto_traces, scatter(
            x = [record[:active_groups] for record in front], y = [record[:validation_matching] for record in front],
            mode = "lines+markers", name = "$(uppercase(string(method))) pooled front",
            line = attr(color = method === :go ? "#000000" : "#7B3294", width = 3), marker = attr(size = 7),
        ))
        for required in 1:3
            rows = filter(row -> row.method === method && row.attained_seeds == required && isfinite(row.validation_mse), metrics.global_attainment_rows)
            push!(pareto_traces, scatter(
                x = [row.active_groups for row in rows], y = [row.validation_mse for row in rows], mode = "lines",
                name = "$(uppercase(string(method))) attainment $required/3", line = attr(color = method === :go ? "#555555" : "#B358A0", width = 1.4, dash = ("dot", "dash", "solid")[required]),
            ))
        end
    end
    pareto_plot = Plot(pareto_traces, Layout(template = "plotly_white", width = 950, height = 600, title = "Package 6 $(options.protocol): native SC Pareto fronts", xaxis = attr(title = "Active SC groups (lower is sparser)"), yaxis = attr(title = "Autoregressive validation MSE", type = "log")))
    paths[:pareto] = save_svg(pareto_plot, joinpath(output, "pareto.svg"); width = 950, height = 600)

    group_traces = PlotlyJS.GenericTrace[]
    mse_traces = PlotlyJS.GenericTrace[]
    proximity_traces = PlotlyJS.GenericTrace[]
    archive_traces = PlotlyJS.GenericTrace[]
    trajectory_records = Dict{String, Any}()
    for run in audit.runs
        records = sort(run.records; by = record -> record[:update])
        front_ids = Set(string(record[:candidate_id]) for record in metrics.run_fronts[run.job.id])
        reset_updates = Set(row.update for row in metrics.reset_event_rows if row.run_id == run.job.id)
        preserve = [index for (index, record) in enumerate(records) if string(record[:candidate_id]) in front_ids || record[:update] in reset_updates]
        indices = downsample_indices(records; maximum_points = 700, preserve)
        sampled = records[indices]
        trajectory_records[run.job.id] = sampled
        trace_line = attr(color = strength_color(run.job.strength_index), width = run.job.method === :go ? 1.5 : 2.2)
        push!(group_traces, scatter(x = [r[:update] for r in sampled], y = [r[:active_groups] for r in sampled], mode = "lines", name = run.job.id, legendgroup = run.job.id, showlegend = run.job.replicate == 1, line = trace_line))
        push!(mse_traces, scatter(x = [r[:update] for r in sampled], y = [r[:validation_matching] for r in sampled], mode = "lines", name = run.job.id, legendgroup = run.job.id, showlegend = run.job.replicate == 1, line = trace_line))
        measured = metrics.checkpoints[run.job.id]
        measured_sample = measured[downsample_indices(measured; maximum_points = 700)]
        push!(proximity_traces, scatter(x = [r[:update] for r in measured_sample], y = [min(r[:strength_front_regret], 5.0) for r in measured_sample], mode = "lines", name = run.job.id, legendgroup = run.job.id, showlegend = run.job.replicate == 1, line = trace_line))
        convergence = filter(row -> row.run_id == run.job.id, metrics.archive_rows)
        convergence_sample = convergence[downsample_indices(convergence; maximum_points = 700)]
        push!(archive_traces, scatter(x = [r.update for r in convergence_sample], y = [r.coverage for r in convergence_sample], mode = "lines", name = run.job.id, legendgroup = run.job.id, showlegend = run.job.replicate == 1, line = trace_line))
    end
    paths[:active_group_trajectories] = save_svg(Plot(group_traces, Layout(template = "plotly_white", title = "Active-group trajectories", xaxis = attr(title = "Training update"), yaxis = attr(title = "Active SC groups"))), joinpath(output, "active_group_trajectories.svg"))
    paths[:mse_trajectories] = save_svg(Plot(mse_traces, Layout(template = "plotly_white", title = "Validation trajectories", xaxis = attr(title = "Training update"), yaxis = attr(title = "Validation MSE", type = "log"))), joinpath(output, "validation_mse_trajectories.svg"))
    paths[:front_proximity] = save_svg(Plot(proximity_traces, Layout(template = "plotly_white", title = "Regret to pooled strength front (clipped at 5)", xaxis = attr(title = "Training update"), yaxis = attr(title = "Relative front regret"), shapes = [attr(type = "line", x0 = 0, x1 = options.protocol === :fixed ? 35000 : 50000, y0 = 0.10, y1 = 0.10, line = attr(dash = "dash", color = "black"))])), joinpath(output, "front_proximity_excursions.svg"))
    paths[:archive_convergence] = save_svg(Plot(archive_traces, Layout(template = "plotly_white", title = "Monotone archive convergence", xaxis = attr(title = "Training update"), yaxis = attr(title = "Final-front envelope coverage", range = [0, 1.02]))), joinpath(output, "archive_convergence.svg"))

    hitting_traces = PlotlyJS.GenericTrace[]
    for strength_index in 1:5
        rows = filter(row -> row.method === :go && row.strength_index == strength_index && row.target_groups in HIGHLIGHT_TARGETS && row.reachable, metrics.hitting_rows)
        targets = collect(HIGHLIGHT_TARGETS)
        medians = [begin values = [Float64(row.first_update) for row in rows if row.target_groups == target]; isempty(values) ? NaN : median(values) end for target in targets]
        push!(hitting_traces, scatter(x = targets, y = medians, mode = "lines+markers", name = @sprintf("GO λ=%.4g", strengths[strength_index]), line = attr(color = strength_color(strength_index))))
    end
    gr_rows = filter(row -> row.method === :gr && row.target_groups in HIGHLIGHT_TARGETS && row.reachable, metrics.hitting_rows)
    gr_targets = collect(HIGHLIGHT_TARGETS)
    gr_medians = [begin values = [Float64(row.first_update) for row in gr_rows if row.target_groups == target]; isempty(values) ? NaN : median(values) end for target in gr_targets]
    push!(hitting_traces, scatter(x = gr_targets, y = gr_medians, mode = "lines+markers", name = "GR reference", line = attr(color = "#7B3294", width = 2.5, dash = "dash")))
    paths[:hitting_times] = save_svg(Plot(hitting_traces, Layout(template = "plotly_white", title = "Median first hitting times", xaxis = attr(title = "Target active SC groups", autorange = "reversed"), yaxis = attr(title = "Training update"))), joinpath(output, "hitting_times.svg"))

    for method in (:go, :gr)
        mask_result = metrics.masks[method]
        run_ids = sort!([run.job.id for run in audit.runs if run.job.method === method])
        matrix = fill(NaN, length(run_ids), length(run_ids))
        for index in eachindex(run_ids); matrix[index, index] = 1.0; end
        pair_values = Dict{Tuple{Int, Int}, Vector{Float64}}()
        for row in mask_result.pair_rows
            left, right = findfirst(==(row.left_run), run_ids), findfirst(==(row.right_run), run_ids)
            push!(get!(pair_values, (left, right), Float64[]), row.jaccard)
        end
        for ((left, right), values) in pair_values
            matrix[left, right] = matrix[right, left] = mean(values)
        end
        heatmap_plot = Plot(heatmap(z = matrix, x = run_ids, y = run_ids, zmin = 0, zmax = 1, colorscale = "Viridis"), Layout(template = "plotly_white", title = "$(uppercase(string(method))) front-mask Jaccard at shared group counts"))
        paths[Symbol("$(method)_jaccard")] = save_svg(heatmap_plot, joinpath(output, "$(method)_mask_jaccard.svg"); width = 900, height = 760)
        frequency = mask_result.selection_frequency
        if !isempty(frequency)
            channel_rows = length(frequency) % 3 == 0 ? reshape(frequency, 3, :) : reshape(frequency, 1, :)
            frequency_plot = Plot(heatmap(z = channel_rows, zmin = 0, zmax = 1, colorscale = "Blues"), Layout(template = "plotly_white", title = "$(uppercase(string(method))) selection frequency: sparsest per-run mask with MSE ≤ $(quality_threshold)", xaxis = attr(title = "Global sensor position"), yaxis = attr(title = "Channel")))
            paths[Symbol("$(method)_selection_map")] = save_svg(frequency_plot, joinpath(output, "$(method)_selection_frequency.svg"); width = 1000, height = 420)
        end
    end

    traces3d = PlotlyJS.GenericTrace[]
    for run in audit.runs
        sampled = trajectory_records[run.job.id]
        push!(traces3d, scatter3d(
            x = [record[:active_groups] for record in sampled],
            y = [log10(max(record[:validation_matching], eps())) for record in sampled],
            z = [record[:update] for record in sampled],
            mode = "lines", name = run.job.id, line = attr(color = strength_color(run.job.strength_index), width = 3),
        ))
    end
    plot3d = Plot(traces3d, Layout(title = "Package 6 $(options.protocol) training trajectories", scene = attr(xaxis = attr(title = "Active SC groups"), yaxis = attr(title = "log10 validation MSE"), zaxis = attr(title = "Update"))))
    html_path = joinpath(output, "trajectories_3d.html")
    open(html_path, "w") do io
        show(io, MIME"text/html"(), plot3d)
    end
    paths[:interactive_3d] = html_path
    return paths
end

function freeze_candidate_manifest(options, audit, metrics)
    all_go = reduce(vcat, [run.records for run in audit.runs if run.job.method === :go])
    quality_threshold = P6_QUALITY_THRESHOLDS[options.protocol]
    match, sparse, _ = select_test_candidates(all_go; mse_threshold = quality_threshold)
    selected = Dict{Symbol, Any}[]
    for (role, candidate) in ((:C_match, match), (:C_sparse, sparse))
        isnothing(candidate) && continue
        selected_candidate = copy(candidate)
        selected_candidate[:selection_role] = role
        checkpoint = candidate_checkpoint_for_record(
            string(selected_candidate[:source_run_directory]),
            selected_candidate,
        )
        isfile(checkpoint) || error("Selected candidate $(candidate[:candidate_id]) has no loadable checkpoint.")
        push!(selected, selected_candidate)
    end
    path = joinpath(analysis_directory(options), "candidate_manifest.jld2")
    expected_ids = string.(getindex.(selected, :candidate_id))
    if isfile(path)
        loaded = JLD2.load(path)
        Bool(loaded["frozen_before_test"]) || error("Existing candidate manifest is not frozen.")
        existing = [Dict{Symbol, Any}(Symbol(key) => value for (key, value) in raw) for raw in loaded["candidates"]]
        string.(getindex.(existing, :candidate_id)) == expected_ids || error("Frozen candidate selection differs from recomputed validation-only selection.")
        return path
    end
    expert_paths = unique(string(run.config[:expert_path]) for run in audit.runs)
    length(expert_paths) == 1 || error("Runs record different expert paths.")
    atomic_save(
        path;
        schema_version = P6_SCHEMA_VERSION,
        experiment = :package6_sc_go_sensitivity,
        protocol = options.protocol,
        selection_source = :pooled_native_go_validation_front,
        selection_uses_test_data = false,
        sparse_mse_threshold = quality_threshold,
        tie_breaker = :lower_mse_then_earlier_update_then_lexicographic_run_id,
        candidates = selected,
        expert_identifier = audit.expert_identifier,
        expert_path = only(expert_paths),
        frozen_before_test = true,
        frozen_at = string(Dates.now()),
    )
    return path
end

function run_terminal_test(options, manifest_path)
    options.skip_test && return nothing
    command = `$(Base.julia_cmd()) --startup-file=no --project=$PROJECT_ROOT $(joinpath(@__DIR__, "run_study_test_worker.jl")) --protocol $(string(options.protocol)) --results-dir $(options.results_root) --manifest $manifest_path`
    options.parallel_test && (command = `$command --parallel-test`)
    run(command)
    result = joinpath(analysis_directory(options), "test", "test_results.jld2")
    isfile(result) || error("Terminal test worker exited without $result.")
    return result
end

function write_report(options, audit, metrics, paths, manifest_path, test_result)
    path = joinpath(analysis_directory(options), "report.md")
    selected = JLD2.load(manifest_path)["candidates"]
    open(path, "w") do io
        println(io, "# Package 6 SC-GO sensitivity and stability study — $(uppercasefirst(string(options.protocol))) IC\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "## Scope and frozen protocol\n")
        println(io, "This report covers Separate-Channel grouping only. It audits 15 GO runs (five strengths × three paired replicates) and three paired GR reference runs. All runs use native sparsity, regression learning rate `2e-4`, validation every 25 updates from update 0, no result-dependent stopping, no fine-tuning, and no hard thresholding. The technical strength calibration is not part of this scientific study.\n")
        println(io, "- GO strengths: `$(join(P6_STRENGTHS[options.protocol], "`, `"))`")
        println(io, "- GR reference strength: `$(P6_GR_STRENGTH[options.protocol])`")
        println(io, "- Sparse-candidate quality threshold: `$(P6_QUALITY_THRESHOLDS[options.protocol])`")
        println(io, "- Training updates: `$(P6_UPDATES[options.protocol])`")
        println(io, "- Training/validation batch sizes: `$(P6_TRAINING_BATCH_SIZE[options.protocol])` / `$(P6_VALIDATION_BATCH_SIZE[options.protocol])`")
        println(io, "- Master seed: `$(P6_MASTER_SEED)`; replicate seed pairs: `$(join(("r$(r)=$(seed_plan(r).apprentice_seed)/$(seed_plan(r).batch_seed)" for r in 1:3), ", "))`\n")
        println(io, "## Audit\n")
        println(io, "PASS. Exactly 15 GO and 3 GR runs were complete. Configuration fingerprints, expert and corpus identities, update budgets, evaluation coverage, paired apprentice/batch seeds, and initial apprentice parameter hashes were consistent. Test data did not enter training or candidate selection.\n")
        println(io, "## Run summary\n")
        println(io, "| Run | Method | λ | Replicate | Front size | Best MSE | Sparsest groups | Late-20% front-near | Runtime (s) |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sort(metrics.run_summary_rows; by = row -> row.run_id)
            @printf(io, "| `%s` | %s | %.6g | %d | %d | %.4e | %d | %.3f | %.1f |\n", row.run_id, uppercase(string(row.method)), row.strength, row.replicate, row.front_size, row.best_validation_mse, row.sparsest_groups, row.late20_front_near_fraction, row.runtime_seconds)
        end
        println(io, "\n## Stability definitions and late-training results\n")
        println(io, "Front regret is relative excess MSE above the final envelope at the same or a sparser group count. A checkpoint is front-near when its MSE is at most 1.10 times the pooled strength-front envelope. Excursions are consecutive non-near checkpoint blocks; recovery is measured in optimizer updates. The primary late window is the final 20%, with 10% and 30% robustness windows retained in the CSV/JLD2 output.\n")
        println(io, "| Run | Near fraction | Median regret | P90 regret | Excursions | Median recovery | Unresolved end |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        for row in sort(filter(row -> row.window_fraction == 0.20, metrics.late_rows); by = row -> row.run_id)
            @printf(io, "| `%s` | %.3f | %.4f | %.4f | %d | %s | %d |\n", row.run_id, row.front_near_fraction, row.median_strength_front_regret, row.p90_strength_front_regret, row.excursion_count, isfinite(row.median_recovery_updates) ? @sprintf("%.1f", row.median_recovery_updates) : "NA", row.unresolved_end_excursions)
        end
        println(io, "\n## Reachability and hitting times\n")
        println(io, "A target is reached when a checkpoint has no more than the target number of active SC groups. Validation MSE is recorded at the first hit; non-reached targets remain explicit rather than being extrapolated.\n")
        println(io, "| Method | λ | Target groups | Reachability | Median first update | Median MSE at hit |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for row in sort(filter(row -> row.target_groups in HIGHLIGHT_TARGETS, metrics.hitting_summary_rows); by = row -> (string(row.method), row.strength, -row.target_groups))
            update_text = row.median_first_update === missing ? "NA" : @sprintf("%.0f", row.median_first_update)
            mse_text = row.median_validation_mse_at_hit === missing ? "NA" : @sprintf("%.4e", row.median_validation_mse_at_hit)
            @printf(io, "| %s | %.6g | %d | %.2f | %s | %s |\n", uppercase(string(row.method)), row.strength, row.target_groups, row.reachability_rate, update_text, mse_text)
        end
        println(io, "\n## Resets and archive convergence\n")
        println(io, "A group reset increases the active-group count; an MSE reset increases validation MSE by more than 10% relative to the preceding evaluation. Archive coverage is monotone and measures the fraction of final-front group envelopes already reached within 10%.\n")
        println(io, "| Run | Group resets/1k | MSE resets/1k | Joint/1k | Updates to 90% | Updates to 100% |")
        println(io, "|---|---:|---:|---:|---:|---:|")
        for reset in sort(metrics.reset_rows; by = row -> row.run_id)
            convergence = first(filter(row -> row.run_id == reset.run_id, metrics.archive_rows))
            u90 = convergence.updates_to_90 === missing ? "NA" : string(convergence.updates_to_90)
            u100 = convergence.updates_to_100 === missing ? "NA" : string(convergence.updates_to_100)
            @printf(io, "| `%s` | %.3f | %.3f | %.3f | %s | %s |\n", reset.run_id, reset.group_reset_rate_per_1000, reset.mse_reset_rate_per_1000, reset.joint_reset_rate_per_1000, u90, u100)
        end
        println(io, "\n## Strength trends and mask stability\n")
        println(io, "Strength trends are descriptive within paired seeds. Spearman values are reported without significance tests. Exact-count mask comparisons use only pairs of run-front masks with an identical active-group count.\n")
        println(io, "| Replicate | Metric | Spearman ρ |")
        println(io, "|---:|---|---:|")
        for row in metrics.trend_rows
            @printf(io, "| %d | `%s` | %s |\n", row.replicate, string(row.metric), isfinite(row.spearman_rho) ? @sprintf("%.3f", row.spearman_rho) : "NA")
        end
        for method in (:go, :gr)
            similarities = [row.jaccard for row in metrics.masks[method].pair_rows]
            if isempty(similarities)
                println(io, "\n- $(uppercase(string(method))): no exact common front-group counts were available for Jaccard comparison.")
            else
                @printf(io, "\n- %s: %d exact-count mask pairs; median Jaccard %.3f, minimum %.3f, maximum %.3f.\n", uppercase(string(method)), length(similarities), median(similarities), minimum(similarities), maximum(similarities))
            end
        end
        println(io, "\n## Validation-only candidate freeze\n")
        println(io, "The immutable [candidate manifest](candidate_manifest.jld2) was written before any terminal rollout. `C_match` minimizes validation MSE on the pooled native GO front. `C_sparse`, when distinct and available, minimizes active SC groups subject to validation MSE ≤ $(P6_QUALITY_THRESHOLDS[options.protocol]). Test results cannot change this manifest.\n")
        for raw in selected
            candidate = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in raw)
            @printf(io, "- `%s`: run `%s`, λ=%.6g, update %d, %d active groups, validation MSE %.6e.\n", string(candidate[:selection_role]), string(candidate[:run_id]), Float64(candidate[:regularization_strength]), Int(candidate[:update]), Int(candidate[:active_groups]), Float64(candidate[:validation_matching]))
        end
        println(io, "\n## Outputs\n")
        for (name, plot_path) in sort!(collect(paths); by = pair -> string(first(pair)))
            relative = relpath(string(plot_path), dirname(path))
            println(io, "- [$(replace(string(name), '_' => ' '))]($(replace(relative, '\\' => '/')))")
        end
        println(io, "- [Full machine-readable metrics](metrics.jld2)")
        println(io, "- [CSV metric tables](csv/)")
        if isnothing(test_result)
            println(io, "\n## Terminal test\n\nNot run (`--skip-test`). The frozen selection is unchanged.")
        else
            loaded = JLD2.load(test_result)
            println(io, "\n## Terminal test (selection-inert)\n")
            expert_source = haskey(loaded, "expert_source") ? Symbol(loaded["expert_source"]) : :terminal_rollout
            source_text = expert_source === :baseline_artifact ?
                "The expert values were reused from the matching Revision/Baselines artifact; the frozen GO candidate(s) were rolled out here." :
                "The expert and frozen GO candidate(s) were rolled out here."
            println(io, "The Fixed protocol uses the shared 200-step episode; the Varying protocol uses all eight predeclared test episodes. $source_text These returns were not used for selection or training.\n")
            for summary in loaded["summaries"]
                @printf(io, "- `%s` (%s): mean 200-step return %.6f.\n", string(summary.id), string(summary.role), Float64(summary.mean_return))
            end
            println(io, "- [Test reward curves](test/test_reward_curves.svg)")
            options.protocol === :varying && println(io, "- [Varying return boxplot](test/test_return_boxplot.svg)")
            println(io, "- [Test episode CSV](test/test_episodes.csv)")
            println(io, "- [Per-case test returns](test/test_returns.csv)")
        end
        println(io, "\n## Interpretation boundary\n")
        println(io, "Strength effects are reported descriptively and with within-replicate Spearman rank correlations. With three replicates, no inferential significance claim is made. GR is an offline stability reference only and never participates in closed-loop candidate selection. Package 7/8 will add GC, SC thresholding, and further regularizers.")
    end
    return path
end

function persist_metrics(options, audit, metrics)
    directory = analysis_directory(options)
    csv_directory = joinpath(directory, "csv")
    write_csv(joinpath(csv_directory, "run_summary.csv"), metrics.run_summary_rows)
    write_csv(joinpath(csv_directory, "late_windows.csv"), metrics.late_rows)
    write_csv(joinpath(csv_directory, "hitting_times.csv"), metrics.hitting_rows)
    write_csv(joinpath(csv_directory, "hitting_summary.csv"), metrics.hitting_summary_rows)
    write_csv(joinpath(csv_directory, "reset_summary.csv"), metrics.reset_rows)
    write_csv(joinpath(csv_directory, "reset_events.csv"), metrics.reset_event_rows)
    write_csv(joinpath(csv_directory, "archive_convergence.csv"), metrics.archive_rows)
    write_csv(joinpath(csv_directory, "attainment.csv"), metrics.attainment_rows)
    write_csv(joinpath(csv_directory, "global_attainment.csv"), metrics.global_attainment_rows)
    write_csv(joinpath(csv_directory, "strength_trends.csv"), metrics.trend_rows)
    write_csv(joinpath(csv_directory, "go_mask_jaccard.csv"), metrics.masks[:go].pair_rows)
    write_csv(joinpath(csv_directory, "gr_mask_jaccard.csv"), metrics.masks[:gr].pair_rows)
    selection_frequency_rows = NamedTuple[]
    for method in (:go, :gr), (index, frequency) in enumerate(metrics.masks[method].selection_frequency)
        push!(selection_frequency_rows, (method, global_mask_index = index, selection_frequency = frequency))
    end
    write_csv(joinpath(csv_directory, "mask_selection_frequency.csv"), selection_frequency_rows)
    front_rows = Dict{Symbol, Any}[]
    front_fields(record) = Dict{Symbol, Any}(
        key => record[key] for key in (:run_id, :method, :strength_index, :regularization_strength, :replicate, :update, :active_groups, :validation_matching, :candidate_id)
        if haskey(record, key)
    )
    for (run_id, front) in metrics.run_fronts, record in front
        push!(front_rows, merge(front_fields(record), Dict(:front_scope => :run, :front_id => run_id)))
    end
    for ((method, strength_index), front) in metrics.strength_fronts, record in front
        push!(front_rows, merge(front_fields(record), Dict(:front_scope => :strength, :front_id => "$(method)_s$(strength_index)")))
    end
    for (method, front) in metrics.global_fronts, record in front
        push!(front_rows, merge(front_fields(record), Dict(:front_scope => :global_method, :front_id => string(method))))
    end
    write_csv(joinpath(csv_directory, "front_points.csv"), front_rows)
    checkpoint_rows = [row for rows in values(metrics.checkpoints) for row in rows]
    checkpoint_fields = (
        :run_id,
        :method,
        :strength_index,
        :regularization_strength,
        :replicate,
        :update,
        :active_groups,
        :validation_matching,
        :own_front_regret,
        :strength_front_regret,
        :front_near,
        :candidate_id,
    )
    compact_checkpoint_rows = [
        NamedTuple{checkpoint_fields}(Tuple(row[key] for key in checkpoint_fields))
        for row in checkpoint_rows
    ]
    write_csv(joinpath(csv_directory, "checkpoint_metrics.csv"), compact_checkpoint_rows)
    path = joinpath(directory, "metrics.jld2")
    atomic_save(
        path;
        schema_version = P6_SCHEMA_VERSION,
        experiment = :package6_sc_go_sensitivity,
        protocol = options.protocol,
        quality_threshold = P6_QUALITY_THRESHOLDS[options.protocol],
        generated_at = string(Dates.now()),
        run_fronts = metrics.run_fronts,
        strength_fronts = metrics.strength_fronts,
        global_fronts = metrics.global_fronts,
        checkpoint_rows = compact_checkpoint_rows,
        late_rows = metrics.late_rows,
        hitting_rows = metrics.hitting_rows,
        hitting_summary_rows = metrics.hitting_summary_rows,
        reset_rows = metrics.reset_rows,
        reset_event_rows = metrics.reset_event_rows,
        archive_rows = metrics.archive_rows,
        run_summary_rows = metrics.run_summary_rows,
        attainment_rows = metrics.attainment_rows,
        global_attainment_rows = metrics.global_attainment_rows,
        trend_rows = metrics.trend_rows,
        masks = metrics.masks,
        expert_identifier = audit.expert_identifier,
    )
    return path
end

function analyze_study(options)
    jobs = wait_for_runs(options)
    audit = audit_runs(options, jobs)
    metrics = build_metrics(audit)
    persist_metrics(options, audit, metrics)
    paths = make_plots(options, audit, metrics)
    manifest = freeze_candidate_manifest(options, audit, metrics)
    test_result = run_terminal_test(options, manifest)
    report = write_report(options, audit, metrics, paths, manifest, test_result)
    failure_report = joinpath(analysis_directory(options), "analysis_failure_report.md")
    isfile(failure_report) && rm(failure_report)
    write_status!(
        joinpath(analysis_directory(options), "status.jld2");
        state = :complete,
        protocol = options.protocol,
        training_run_count = length(jobs),
        candidate_manifest = manifest,
        report,
        terminal_test_result = test_result,
        completed_at = string(Dates.now()),
    )
    println("Package 6 $(options.protocol) analysis complete: $report")
    return report
end

function record_analysis_failure(options, error_value, backtrace_value)
    directory = analysis_directory(options)
    mkpath(directory)
    message = sprint(showerror, error_value)
    backtrace_text = sprint(Base.show_backtrace, backtrace_value)
    report = joinpath(directory, "analysis_failure_report.md")
    open(report, "w") do io
        println(io, "# Package 6 $(options.protocol) analysis failure\n")
        println(io, "Generated: $(Dates.now())\n")
        println(io, "```text")
        println(io, message)
        println(io, backtrace_text)
        println(io, "```")
    end
    write_status!(
        joinpath(directory, "status.jld2");
        state = :failed,
        protocol = options.protocol,
        error_type = string(typeof(error_value)),
        error_message = message,
        backtrace = backtrace_text,
        failure_report = report,
        failed_at = string(Dates.now()),
    )
    return report
end

function analysis_worker_main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    try
        return analyze_study(options)
    catch error_value
        backtrace_value = catch_backtrace()
        record_analysis_failure(options, error_value, backtrace_value)
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        analysis_worker_main()
    catch error_value
        backtrace_value = catch_backtrace()
        Base.display_error(stderr, error_value, backtrace_value)
        exit(1)
    end
end
