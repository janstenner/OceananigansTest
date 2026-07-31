ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Dates
using JLD2
using Plots
using Printf
using Statistics

include(joinpath(@__DIR__, "MATIPPOExperiment.jl"))
using .MATIPPOExperiment

const COLORS = Dict(:mat => :royalblue3, :ippo => :darkorange2)
const LABELS = Dict(:mat => "MAT", :ippo => "IPPO")
const WINDOW = 50

csv_value(value) = begin
    text = replace(string(value), '"' => "\"\"")
    occursin(r"[,\n\"]", text) ? "\"$text\"" : text
end

function write_csv(path, columns, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, column)) for column in columns), ','))
        end
    end
    return path
end

function optional_read(file, key, default)
    return haskey(file, key) ? read(file, key) : default
end

function read_training(path)
    return try
        JLD2.jldopen(path, "r") do file
            status = string(read(file, "status"))
            protocol = Symbol(read(file, "protocol"))
            algorithm = haskey(file, "algorithm") ? Symbol(read(file, "algorithm")) : :mat
            run_id = string(optional_read(file, "run_id", optional_read(file, "package4_run_id", basename(dirname(dirname(path))))))
            rewards = Float64.(optional_read(file, "rewards", Float64[]))
            (
                path = path,
                status = status,
                run_id = run_id,
                protocol = protocol,
                algorithm = algorithm,
                origin = Symbol(optional_read(file, "origin", optional_read(file, "package4_origin", :unknown))),
                run_seed = Int(read(file, "run_seed")),
                ic_seed = Int(read(file, "ic_seed")),
                episode_target = Int(read(file, "episode_target")),
                episodes_completed = Int(optional_read(file, "episodes_completed", length(rewards))),
                elapsed_seconds = Float64(optional_read(file, "elapsed_seconds", NaN)),
                rewards = rewards,
                errored_episodes = Int.(optional_read(file, "errored_episodes", Int[])),
                error_message = string(optional_read(file, "error_message", "")),
                parameter_count = Int(optional_read(file, "parameter_count", optional_read(file, "full_parameter_count", -1))),
                trace = optional_read(file, "initial_condition_trace", NamedTuple[]),
                imported_source = string(optional_read(file, "package4_source_path", "")),
            )
        end
    catch error_value
        @warn "Could not read result" path exception = error_value
        nothing
    end
end

function read_validation(path)
    isfile(path) || return nothing
    return try
        JLD2.jldopen(path, "r") do file
            string(read(file, "status")) == "complete" || return nothing
            (
                mean = Float64(read(file, "validation_mean")),
                scores = Float64.(read(file, "validation_scores")),
                cases = read(file, "validation_cases"),
            )
        end
    catch error_value
        @warn "Could not read validation" path exception = error_value
        nothing
    end
end

function scan_results(results_directory)
    records = NamedTuple[]
    failures = NamedTuple[]
    plan_entries = if isfile(joinpath(results_directory, "run_plan.jld2"))
        Dict(entry.run_id => entry for entry in collect_plan_entries(results_directory))
    else
        Dict{String, Any}()
    end
    runs_root = joinpath(results_directory, "runs")
    isdir(runs_root) || return records, failures
    for (directory, _, files) in walkdir(runs_root)
        for filename in sort(files)
            endswith(filename, ".jld2") || continue
            endswith(filename, "_validation.jld2") && continue
            filename in ("mat.jld2", "ippo.jld2") || continue
            training = read_training(joinpath(directory, filename))
            isnothing(training) && continue
            if training.status != "complete"
                push!(failures, training)
                continue
            end
            expected = MATIPPOExperiment.DEFAULT_EPISODES[training.protocol]
            if training.episode_target != expected ||
               training.episodes_completed != expected ||
               length(training.rewards) != expected
                push!(failures, merge(training, (
                    status = "invalid",
                    error_message = "Expected $expected episodes and rewards.",
                )))
                continue
            end
            if haskey(plan_entries, training.run_id)
                entry = plan_entries[training.run_id]
                training.run_seed == entry.run_seed || error("Plan run-seed mismatch for $(training.path).")
                training.ic_seed == entry.ic_seed || error("Plan IC-seed mismatch for $(training.path).")
                if training.protocol === :varying
                    observed = MATIPPOExperiment.normalize_trace(training.trace)
                    observed == entry.varying_trace || error(
                        "Initial-condition trace differs from the plan in $(training.path).",
                    )
                end
            end
            validation_file = replace(training.path, r"\.jld2$" => "_validation.jld2")
            validation = read_validation(validation_file)
            if isfile(validation_file) && isnothing(validation)
                message = try
                    JLD2.jldopen(validation_file, "r") do file
                        string(optional_read(file, "error_message", "Invalid validation sidecar."))
                    end
                catch error_value
                    sprint(showerror, error_value)
                end
                push!(failures, merge(training, (
                    path = validation_file,
                    status = "validation_failed",
                    error_message = message,
                )))
            end
            push!(records, merge(training, (
                validation_path = validation_file,
                validation_mean = isnothing(validation) ? missing : validation.mean,
                validation_scores = isnothing(validation) ? Float64[] : validation.scores,
                validation_cases = isnothing(validation) ? Any[] : validation.cases,
            )))
        end
    end
    sort!(records; by = item -> (item.protocol, item.run_id, item.algorithm))
    return records, failures
end

rolling_mean(values, width = WINDOW) = length(values) < width ? Float64[] : [
    mean(@view values[(index - width + 1):index]) for index in width:length(values)
]

function curve_statistics(records)
    rows = NamedTuple[]
    for protocol in (:fixed, :varying), algorithm in (:mat, :ippo)
        curves = [rolling_mean(record.rewards) for record in records if
                  record.protocol == protocol && record.algorithm == algorithm]
        filter!(curve -> !isempty(curve), curves)
        isempty(curves) && continue
        count = minimum(length, curves)
        matrix = reduce(hcat, (curve[1:count] for curve in curves))
        for index in 1:count
            values = vec(matrix[index, :])
            center = mean(values)
            lower_values = center .- values[values .< center]
            upper_values = values[values .> center] .- center
            push!(rows, (
                protocol = protocol,
                algorithm = algorithm,
                episode = index + WINDOW - 1,
                n = length(values),
                mean = center,
                std = std(values; corrected = length(values) > 1),
                median = median(values),
                q25 = quantile(values, 0.25),
                q75 = quantile(values, 0.75),
                mean_deviation_lower = isempty(lower_values) ? 0.0 : mean(lower_values),
                mean_deviation_upper = isempty(upper_values) ? 0.0 : mean(upper_values),
            ))
        end
    end
    return rows
end

function paired_curve_data(records)
    curves = NamedTuple[]
    statistics = NamedTuple[]
    for protocol in (:fixed, :varying)
        ids = sort!(unique(record.run_id for record in records if record.protocol == protocol))
        for id in ids
            mat = [record for record in records if record.protocol == protocol &&
                   record.run_id == id && record.algorithm == :mat]
            ippo = [record for record in records if record.protocol == protocol &&
                    record.run_id == id && record.algorithm == :ippo]
            length(mat) == 1 && length(ippo) == 1 || continue
            mat_curve = rolling_mean(only(mat).rewards)
            ippo_curve = rolling_mean(only(ippo).rewards)
            count = min(length(mat_curve), length(ippo_curve))
            count == 0 && continue
            push!(curves, (
                protocol = protocol,
                run_id = id,
                values = mat_curve[1:count] .- ippo_curve[1:count],
            ))
        end
        subset = [curve for curve in curves if curve.protocol == protocol]
        isempty(subset) && continue
        count = minimum(length(curve.values) for curve in subset)
        matrix = reduce(hcat, (curve.values[1:count] for curve in subset))
        for index in 1:count
            values = vec(matrix[index, :])
            push!(statistics, (
                protocol = protocol,
                episode = index + WINDOW - 1,
                n = length(values),
                mean = mean(values),
                median = median(values),
                q25 = quantile(values, 0.25),
                q75 = quantile(values, 0.75),
            ))
        end
    end
    return curves, statistics
end

function save_both(plot, stem)
    savefig(plot, stem * ".png")
    savefig(plot, stem * ".pdf")
end

function plot_learning_curves(records, stats, plot_directory)
    for protocol in (:fixed, :varying)
        available = [record for record in records if record.protocol == protocol && length(record.rewards) >= WINDOW]
        isempty(available) && continue
        plot_object = plot(
            xlabel = "Episode",
            ylabel = "Score (rolling mean, window=$WINDOW)",
            title = "$(uppercasefirst(string(protocol))) IC learning curves",
            legend = :bottomright,
        )
        for algorithm in (:mat, :ippo)
            subset = [record for record in available if record.algorithm == algorithm]
            for (index, record) in enumerate(subset)
                curve = rolling_mean(record.rewards)
                plot!(
                    plot_object,
                    WINDOW:(WINDOW + length(curve) - 1),
                    curve;
                    color = COLORS[algorithm],
                    alpha = 0.16,
                    linewidth = 1,
                    label = index == 1 ? "$(LABELS[algorithm]) runs (n=$(length(subset)))" : "",
                )
            end
            aggregate = [row for row in stats if row.protocol == protocol && row.algorithm == algorithm]
            isempty(aggregate) && continue
            episodes = getproperty.(aggregate, :episode)
            medians = getproperty.(aggregate, :median)
            q25 = getproperty.(aggregate, :q25)
            q75 = getproperty.(aggregate, :q75)
            means = getproperty.(aggregate, :mean)
            plot!(
                plot_object,
                episodes,
                medians;
                ribbon = (medians .- q25, q75 .- medians),
                fillalpha = 0.20,
                color = COLORS[algorithm],
                linewidth = 3,
                label = "$(LABELS[algorithm]) median + IQR",
            )
            plot!(
                plot_object,
                episodes,
                means;
                color = COLORS[algorithm],
                linestyle = :dash,
                linewidth = 2,
                label = "$(LABELS[algorithm]) mean",
            )
        end
        save_both(plot_object, joinpath(plot_directory, "$(protocol)_learning_curves"))

        individual = plot(
            xlabel = "Episode",
            ylabel = "Score (rolling mean, window=$WINDOW)",
            title = "$(uppercasefirst(string(protocol))) IC individual runs",
            legend = :bottomright,
        )
        for algorithm in (:mat, :ippo)
            subset = [record for record in available if record.algorithm == algorithm]
            for (index, record) in enumerate(subset)
                curve = rolling_mean(record.rewards)
                plot!(individual, WINDOW:(WINDOW + length(curve) - 1), curve;
                      color = COLORS[algorithm], alpha = 0.55, linewidth = 1.4,
                      label = index == 1 ? LABELS[algorithm] : "")
            end
        end
        save_both(individual, joinpath(plot_directory, "$(protocol)_individual_curves"))
    end
end

function plot_paired_curves(curves, stats, plot_directory)
    for protocol in (:fixed, :varying)
        subset = [curve for curve in curves if curve.protocol == protocol]
        aggregate = [row for row in stats if row.protocol == protocol]
        isempty(subset) && continue
        plot_object = plot(
            xlabel = "Episode",
            ylabel = "MAT - IPPO rolling-50 score",
            title = "$(uppercasefirst(string(protocol))) IC paired learning differences (n=$(length(subset)))",
            legend = :bottomright,
        )
        hline!(plot_object, [0.0]; color = :black, linestyle = :dash, label = "zero")
        for curve in subset
            plot!(plot_object, WINDOW:(WINDOW + length(curve.values) - 1), curve.values;
                  color = :purple3, alpha = 0.18, linewidth = 1, label = "")
        end
        episodes = getproperty.(aggregate, :episode)
        medians = getproperty.(aggregate, :median)
        q25 = getproperty.(aggregate, :q25)
        q75 = getproperty.(aggregate, :q75)
        plot!(plot_object, episodes, medians;
              ribbon = (medians .- q25, q75 .- medians), fillalpha = 0.22,
              color = :purple3, linewidth = 3, label = "median + IQR")
        plot!(plot_object, episodes, getproperty.(aggregate, :mean);
              color = :purple3, linestyle = :dot, linewidth = 2, label = "mean")
        save_both(plot_object, joinpath(plot_directory, "$(protocol)_paired_learning_differences"))
    end
end

function final_rows(records)
    return [
        (
            run_id = record.run_id,
            protocol = record.protocol,
            algorithm = record.algorithm,
            origin = record.origin,
            run_seed = record.run_seed,
            ic_seed = record.ic_seed,
            episodes = record.episodes_completed,
            final_last100 = isempty(record.rewards) ? NaN : mean(
                record.rewards[max(1, length(record.rewards) - 99):length(record.rewards)],
            ),
            final_episode = isempty(record.rewards) ? NaN : last(record.rewards),
            best_rolling50 = isempty(rolling_mean(record.rewards)) ? NaN : maximum(rolling_mean(record.rewards)),
            best_rolling50_episode = isempty(rolling_mean(record.rewards)) ? 0 :
                argmax(rolling_mean(record.rewards)) + WINDOW - 1,
            validation_mean = record.validation_mean,
            validation_cases = repr(record.validation_cases),
            elapsed_seconds = record.elapsed_seconds,
            errored_episode_count = length(record.errored_episodes),
            errored_episode_fraction = isempty(record.rewards) ? NaN :
                length(record.errored_episodes) / length(record.rewards),
            parameter_count = record.parameter_count,
            result_path = record.path,
            imported_source = record.imported_source,
        ) for record in records
    ]
end

function paired_rows(final)
    rows = NamedTuple[]
    for protocol in (:fixed, :varying)
        ids = sort!(unique(row.run_id for row in final if row.protocol == protocol))
        for id in ids
            mat = [row for row in final if row.protocol == protocol && row.run_id == id && row.algorithm == :mat]
            ippo = [row for row in final if row.protocol == protocol && row.run_id == id && row.algorithm == :ippo]
            length(mat) <= 1 || error("Duplicate MAT result for $id/$protocol.")
            length(ippo) <= 1 || error("Duplicate IPPO result for $id/$protocol.")
            isempty(mat) || isempty(ippo) || begin
                first(mat).run_seed == first(ippo).run_seed || error("Paired run-seed mismatch.")
                first(mat).ic_seed == first(ippo).ic_seed || error("Paired IC-seed mismatch.")
            end
            push!(rows, (
                run_id = id,
                protocol = protocol,
                has_mat = !isempty(mat),
                has_ippo = !isempty(ippo),
                run_seed = !isempty(mat) ? first(mat).run_seed : first(ippo).run_seed,
                ic_seed = !isempty(mat) ? first(mat).ic_seed : first(ippo).ic_seed,
                mat_final_last100 = isempty(mat) ? missing : first(mat).final_last100,
                ippo_final_last100 = isempty(ippo) ? missing : first(ippo).final_last100,
                difference_mat_minus_ippo = isempty(mat) || isempty(ippo) ? missing :
                    first(mat).final_last100 - first(ippo).final_last100,
                mat_validation = isempty(mat) ? missing : first(mat).validation_mean,
                ippo_validation = isempty(ippo) ? missing : first(ippo).validation_mean,
                mat_origin = isempty(mat) ? missing : first(mat).origin,
                ippo_origin = isempty(ippo) ? missing : first(ippo).origin,
                mat_result_path = isempty(mat) ? missing : first(mat).result_path,
                ippo_result_path = isempty(ippo) ? missing : first(ippo).result_path,
                mat_imported_source = isempty(mat) ? missing : first(mat).imported_source,
                ippo_imported_source = isempty(ippo) ? missing : first(ippo).imported_source,
                mat_validation_cases = isempty(mat) ? missing : first(mat).validation_cases,
                ippo_validation_cases = isempty(ippo) ? missing : first(ippo).validation_cases,
            ))
        end
    end
    return rows
end

function scatter_summary(rows, value_name, title, ylabel, stem; paired = false)
    counts = [
        sum(row.algorithm == algorithm && !ismissing(getproperty(row, value_name)) &&
            isfinite(getproperty(row, value_name)) for row in rows)
        for algorithm in (:mat, :ippo)
    ]
    plot_object = plot(
        title = title,
        ylabel = ylabel,
        legend = false,
        xticks = ([1, 2], ["MAT (n=$(counts[1]))", "IPPO (n=$(counts[2]))"]),
    )
    for (position, algorithm) in enumerate((:mat, :ippo))
        values = Float64[getproperty(row, value_name) for row in rows if
                         row.algorithm == algorithm && !ismissing(getproperty(row, value_name)) &&
                         isfinite(getproperty(row, value_name))]
        isempty(values) && continue
        jitter = length(values) == 1 ? [0.0] : collect(range(-0.08, 0.08; length = length(values)))
        scatter!(plot_object, position .+ jitter, values; color = COLORS[algorithm], alpha = 0.8, markersize = 5)
        plot!(plot_object, [position - 0.16, position + 0.16], fill(median(values), 2);
              color = :black, linewidth = 3)
        plot!(plot_object, [position, position], [quantile(values, 0.25), quantile(values, 0.75)];
              color = :black, linewidth = 2)
    end
    save_both(plot_object, stem)
end

function plot_summaries(final, paired, plot_directory)
    for protocol in (:fixed, :varying)
        subset = [row for row in final if row.protocol == protocol]
        isempty(subset) && continue
        scatter_summary(
            subset, :final_last100,
            "$(uppercasefirst(string(protocol))) IC final performance",
            "Mean score over last 100 training episodes",
            joinpath(plot_directory, "$(protocol)_final_last100"),
        )
        paired_complete = [row for row in paired if row.protocol == protocol &&
                           !ismissing(row.mat_final_last100) && !ismissing(row.ippo_final_last100)]
        if !isempty(paired_complete)
            paired_plot = plot(
                title = "$(uppercasefirst(string(protocol))) IC paired final performance (n=$(length(paired_complete)))",
                ylabel = "Mean score over last 100 training episodes",
                xticks = ([1, 2], ["MAT", "IPPO"]),
                legend = false,
            )
            for row in paired_complete
                plot!(paired_plot, [1, 2], [row.mat_final_last100, row.ippo_final_last100];
                      color = :gray45, alpha = 0.45, linewidth = 1.4)
                scatter!(paired_plot, [1, 2], [row.mat_final_last100, row.ippo_final_last100];
                         color = [COLORS[:mat], COLORS[:ippo]], markersize = 5)
            end
            save_both(paired_plot, joinpath(plot_directory, "$(protocol)_final_last100_paired"))
        end
        validation = [row for row in subset if !ismissing(row.validation_mean)]
        isempty(validation) || scatter_summary(
            validation, :validation_mean,
            "$(uppercasefirst(string(protocol))) IC deterministic validation",
            "Validation score",
            joinpath(plot_directory, "$(protocol)_validation_performance"),
        )
        runtime = [row for row in subset if isfinite(row.elapsed_seconds)]
        isempty(runtime) || scatter_summary(
            runtime, :elapsed_seconds,
            "$(uppercasefirst(string(protocol))) IC runtime",
            "Training time (seconds)",
            joinpath(plot_directory, "$(protocol)_runtimes"),
        )
        scatter_summary(
            subset, :errored_episode_count,
            "$(uppercasefirst(string(protocol))) IC errored episodes",
            "Errored episode count",
            joinpath(plot_directory, "$(protocol)_errored_episodes"),
        )

        differences = Float64[row.difference_mat_minus_ippo for row in paired if
                              row.protocol == protocol && !ismissing(row.difference_mat_minus_ippo)]
        isempty(differences) && continue
        paired_plot = scatter(
            1:length(differences), differences;
            xlabel = "Paired seed index",
            ylabel = "MAT - IPPO last-100 score",
            title = "$(uppercasefirst(string(protocol))) IC paired differences",
            color = :purple3,
            markersize = 6,
            legend = false,
        )
        hline!(paired_plot, [0.0]; color = :black, linestyle = :dash)
        hline!(paired_plot, [median(differences)]; color = :purple3, linewidth = 2)
        save_both(paired_plot, joinpath(plot_directory, "$(protocol)_paired_differences"))
    end
end

function expert_rows(final)
    rows = NamedTuple[]
    for protocol in (:fixed, :varying)
        candidates = [row for row in final if row.protocol == protocol && row.algorithm == :mat &&
                      !ismissing(row.validation_mean)]
        sort!(candidates; by = row -> row.validation_mean, rev = true)
        for (rank, row) in enumerate(candidates)
            push!(rows, (
                protocol = protocol,
                rank = rank,
                run_id = row.run_id,
                validation_mean = row.validation_mean,
                run_seed = row.run_seed,
                ic_seed = row.ic_seed,
                origin = row.origin,
                checkpoint = row.result_path,
            ))
        end
    end
    return rows
end

function diagnostic_rows(final)
    rows = NamedTuple[]
    for protocol in (:fixed, :varying), algorithm in (:mat, :ippo)
        subset = [row for row in final if row.protocol == protocol &&
                  row.algorithm == algorithm && isfinite(row.best_rolling50)]
        sort!(subset; by = row -> row.best_rolling50, rev = true)
        for (rank, row) in enumerate(subset)
            push!(rows, (
                protocol = protocol,
                algorithm = algorithm,
                rank = rank,
                run_id = row.run_id,
                best_rolling50 = row.best_rolling50,
                best_rolling50_episode = row.best_rolling50_episode,
                final_last100 = row.final_last100,
                errored_episode_count = row.errored_episode_count,
            ))
        end
    end
    return rows
end

function main(arguments = ARGS)
    results_directory = MATIPPOExperiment.DEFAULT_RESULTS_DIRECTORY
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument in ("--results-dir", "--results_dir")
            index == length(arguments) && error("Missing value after $argument.")
            results_directory = abspath(arguments[index + 1])
            index += 1
        elseif argument == "--help"
            println("Usage: julia --project=. collect_results.jl [--results-dir PATH]")
            return
        else
            error("Unknown argument '$argument'.")
        end
        index += 1
    end

    records, failures = scan_results(results_directory)
    output_directory = joinpath(results_directory, "analysis")
    plot_directory = joinpath(output_directory, "plots")
    isdir(plot_directory) && rm(plot_directory; recursive = true)
    mkpath(plot_directory)

    stats = curve_statistics(records)
    paired_curves, paired_curve_stats = paired_curve_data(records)
    final = final_rows(records)
    paired = paired_rows(final)
    experts = expert_rows(final)
    diagnostics = diagnostic_rows(final)
    plot_learning_curves(records, stats, plot_directory)
    plot_paired_curves(paired_curves, paired_curve_stats, plot_directory)
    plot_summaries(final, paired, plot_directory)

    write_csv(
        joinpath(output_directory, "curve_statistics.csv"),
        (:protocol, :algorithm, :episode, :n, :mean, :std, :median, :q25, :q75,
         :mean_deviation_lower, :mean_deviation_upper),
        stats,
    )
    write_csv(
        joinpath(output_directory, "paired_curve_statistics.csv"),
        (:protocol, :episode, :n, :mean, :median, :q25, :q75),
        paired_curve_stats,
    )
    write_csv(
        joinpath(output_directory, "run_summary.csv"),
        (:run_id, :protocol, :algorithm, :origin, :run_seed, :ic_seed, :episodes,
         :final_last100, :final_episode, :best_rolling50, :best_rolling50_episode,
         :validation_mean, :validation_cases, :elapsed_seconds, :errored_episode_count,
         :errored_episode_fraction, :parameter_count,
         :result_path, :imported_source),
        final,
    )
    write_csv(
        joinpath(output_directory, "pairing_table.csv"),
        (:run_id, :protocol, :has_mat, :has_ippo, :run_seed, :ic_seed,
         :mat_final_last100, :ippo_final_last100, :difference_mat_minus_ippo,
         :mat_validation, :ippo_validation, :mat_origin, :ippo_origin,
         :mat_result_path, :ippo_result_path, :mat_imported_source,
         :ippo_imported_source, :mat_validation_cases, :ippo_validation_cases),
        paired,
    )
    write_csv(
        joinpath(output_directory, "mat_expert_ranking.csv"),
        (:protocol, :rank, :run_id, :validation_mean, :run_seed, :ic_seed, :origin, :checkpoint),
        experts,
    )
    failure_rows = [(
        path = row.path,
        run_id = row.run_id,
        protocol = row.protocol,
        algorithm = row.algorithm,
        run_seed = row.run_seed,
        ic_seed = row.ic_seed,
        episodes_completed = row.episodes_completed,
        error_message = row.error_message,
    ) for row in failures]
    write_csv(
        joinpath(output_directory, "best_rolling50_diagnostics.csv"),
        (:protocol, :algorithm, :rank, :run_id, :best_rolling50,
         :best_rolling50_episode, :final_last100, :errored_episode_count),
        diagnostics,
    )
    write_csv(
        joinpath(output_directory, "failures.csv"),
        (:path, :run_id, :protocol, :algorithm, :run_seed, :ic_seed,
         :episodes_completed, :error_message),
        failure_rows,
    )

    summary_path = joinpath(output_directory, "collected_results.jld2")
    temporary = summary_path * ".tmp.$(getpid())"
    JLD2.jldsave(
        temporary;
        schema_version = 1,
        collected_at = string(now()),
        results_directory = abspath(results_directory),
        rolling_window = WINDOW,
        records = records,
        failures = failures,
        curve_statistics = stats,
        paired_curve_statistics = paired_curve_stats,
        paired_curves = paired_curves,
        run_summary = final,
        pairing_table = paired,
        mat_expert_ranking = experts,
        best_rolling50_diagnostics = diagnostics,
    )
    mv(temporary, summary_path; force = true)

    println("Collected $(length(records)) complete results and $(length(failures)) failures.")
    for protocol in (:fixed, :varying), algorithm in (:mat, :ippo)
        count = sum(row.protocol == protocol && row.algorithm == algorithm for row in records)
        count > 0 && println("  $protocol/$algorithm: $count")
    end
    println("Analysis written to $output_directory")
    isempty(experts) || println("Top MAT experts are listed in mat_expert_ranking.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
