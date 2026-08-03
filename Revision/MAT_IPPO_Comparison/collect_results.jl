ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Dates
using JLD2
using PlotlyJS
using Printf
using Statistics

include(joinpath(@__DIR__, "MATIPPOExperiment.jl"))
using .MATIPPOExperiment

const COLORS = Dict(:mat => "#277DA1", :ippo => "#F2A13A")
const RUN_COLORS = Dict(
    :mat => "rgba(39, 125, 161, 0.22)",
    :ippo => "rgba(242, 161, 58, 0.22)",
)
const RIBBON_COLORS = Dict(
    :mat => "rgba(39, 125, 161, 0.18)",
    :ippo => "rgba(242, 161, 58, 0.18)",
)
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

function plot_layout(title, xlabel, ylabel; tickvals = nothing, ticktext = nothing,
                     showlegend = true)
    xaxis_options = Dict{Symbol, Any}(
        :title => attr(text = xlabel, standoff = 12),
        :showline => true,
        :mirror => true,
        :linecolor => "#3A3A3A",
        :linewidth => 1,
        :ticks => "outside",
        :gridcolor => "#E6E6E6",
        :zeroline => false,
    )
    if !isnothing(tickvals)
        xaxis_options[:tickmode] = "array"
        xaxis_options[:tickvals] = tickvals
        xaxis_options[:ticktext] = ticktext
        xaxis_options[:range] = [minimum(tickvals) - 0.45, maximum(tickvals) + 0.45]
        xaxis_options[:showgrid] = false
    end
    return Layout(
        template = "plotly_white",
        title = attr(
            text = title,
            x = 0.5,
            xanchor = "center",
            font = attr(size = 22, color = "#252525"),
        ),
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        width = 900,
        height = 560,
        margin = attr(l = 100, r = 30, t = 80, b = 85),
        font = attr(family = "Arial, sans-serif", size = 15, color = "#303030"),
        xaxis = attr(; xaxis_options...),
        yaxis = attr(
            title = attr(text = ylabel, standoff = 12),
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            linewidth = 1,
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        showlegend = showlegend,
        legend = attr(
            x = 0.985,
            y = 0.02,
            xanchor = "right",
            yanchor = "bottom",
            traceorder = "normal",
            bgcolor = "rgba(255, 255, 255, 0.92)",
            bordercolor = "#CFCFCF",
            borderwidth = 1,
            font = attr(size = 13),
        ),
    )
end

function save_svg(plot, stem)
    output = stem * ".svg"
    PlotlyJS.savefig(plot, output; width = 900, height = 560)
    return output
end

function plot_learning_curves(records, stats, plot_directory)
    for protocol in (:fixed, :varying)
        available = [record for record in records if record.protocol == protocol && length(record.rewards) >= WINDOW]
        isempty(available) && continue
        traces = PlotlyJS.GenericTrace[]
        for (algorithm_index, algorithm) in enumerate((:mat, :ippo))
            subset = [record for record in available if record.algorithm == algorithm]
            for (index, record) in enumerate(subset)
                curve = rolling_mean(record.rewards)
                push!(
                    traces,
                    scatter(
                        x = collect(WINDOW:(WINDOW + length(curve) - 1)),
                        y = curve,
                        mode = "lines",
                        name = "$(LABELS[algorithm]) runs (n=$(length(subset)))",
                        legendgroup = "$(algorithm)_runs",
                        legendrank = 10 * algorithm_index,
                        showlegend = index == 1,
                        line = attr(color = RUN_COLORS[algorithm], width = 1),
                        hoverinfo = "skip",
                    ),
                )
            end
            aggregate = [row for row in stats if row.protocol == protocol && row.algorithm == algorithm]
            isempty(aggregate) && continue
            episodes = getproperty.(aggregate, :episode)
            medians = getproperty.(aggregate, :median)
            q25 = getproperty.(aggregate, :q25)
            q75 = getproperty.(aggregate, :q75)
            means = getproperty.(aggregate, :mean)
            push!(
                traces,
                scatter(
                    x = episodes,
                    y = q25,
                    mode = "lines",
                    line = attr(width = 0),
                    hoverinfo = "skip",
                    showlegend = false,
                ),
            )
            push!(
                traces,
                scatter(
                    x = episodes,
                    y = q75,
                    mode = "lines",
                    line = attr(width = 0),
                    fill = "tonexty",
                    fillcolor = RIBBON_COLORS[algorithm],
                    hoverinfo = "skip",
                    showlegend = false,
                ),
            )
            push!(
                traces,
                scatter(
                    x = episodes,
                    y = medians,
                    mode = "lines",
                    name = "$(LABELS[algorithm]) median + IQR",
                    legendrank = 10 * algorithm_index + 1,
                    line = attr(color = COLORS[algorithm], width = 3),
                    hovertemplate = "Episode %{x}<br>Median %{y:.2f}<extra>%{fullData.name}</extra>",
                ),
            )
            push!(
                traces,
                scatter(
                    x = episodes,
                    y = means,
                    mode = "lines",
                    name = "$(LABELS[algorithm]) mean",
                    legendrank = 10 * algorithm_index + 2,
                    line = attr(color = COLORS[algorithm], width = 2, dash = "dash"),
                    hovertemplate = "Episode %{x}<br>Mean %{y:.2f}<extra>%{fullData.name}</extra>",
                ),
            )
        end
        plot_object = Plot(
            traces,
            plot_layout(
                "$(uppercasefirst(string(protocol))) IC learning curves",
                "Episode",
                "Score (rolling mean, window=$WINDOW)",
            ),
        )
        save_svg(plot_object, joinpath(plot_directory, "$(protocol)_learning_curves"))

        individual_traces = PlotlyJS.GenericTrace[]
        for (algorithm_index, algorithm) in enumerate((:mat, :ippo))
            subset = [record for record in available if record.algorithm == algorithm]
            for (index, record) in enumerate(subset)
                curve = rolling_mean(record.rewards)
                push!(
                    individual_traces,
                    scatter(
                        x = collect(WINDOW:(WINDOW + length(curve) - 1)),
                        y = curve,
                        mode = "lines",
                        name = "$(LABELS[algorithm]) (n=$(length(subset)))",
                        legendgroup = string(algorithm),
                        legendrank = algorithm_index,
                        showlegend = index == 1,
                        line = attr(color = COLORS[algorithm], width = 1.5),
                        opacity = 0.55,
                        hovertemplate = "Episode %{x}<br>Score %{y:.2f}<extra>$(LABELS[algorithm])</extra>",
                    ),
                )
            end
        end
        individual = Plot(
            individual_traces,
            plot_layout(
                "$(uppercasefirst(string(protocol))) IC individual runs",
                "Episode",
                "Score (rolling mean, window=$WINDOW)",
            ),
        )
        save_svg(individual, joinpath(plot_directory, "$(protocol)_individual_curves"))
    end
end

function plot_paired_curves(curves, stats, plot_directory)
    for protocol in (:fixed, :varying)
        subset = [curve for curve in curves if curve.protocol == protocol]
        aggregate = [row for row in stats if row.protocol == protocol]
        isempty(subset) && continue
        traces = PlotlyJS.GenericTrace[
            scatter(
                x = [WINDOW, WINDOW + maximum(length(curve.values) for curve in subset) - 1],
                y = [0.0, 0.0],
                mode = "lines",
                name = "zero",
                legendrank = 1,
                line = attr(color = "#303030", width = 1.5, dash = "dash"),
                hoverinfo = "skip",
            ),
        ]
        for curve in subset
            push!(
                traces,
                scatter(
                    x = collect(WINDOW:(WINDOW + length(curve.values) - 1)),
                    y = curve.values,
                    mode = "lines",
                    line = attr(color = RUN_COLORS[:mat], width = 1),
                    hoverinfo = "skip",
                    showlegend = false,
                ),
            )
        end
        episodes = getproperty.(aggregate, :episode)
        medians = getproperty.(aggregate, :median)
        q25 = getproperty.(aggregate, :q25)
        q75 = getproperty.(aggregate, :q75)
        push!(
            traces,
            scatter(
                x = episodes,
                y = q25,
                mode = "lines",
                line = attr(width = 0),
                hoverinfo = "skip",
                showlegend = false,
            ),
        )
        push!(
            traces,
            scatter(
                x = episodes,
                y = q75,
                mode = "lines",
                line = attr(width = 0),
                fill = "tonexty",
                fillcolor = RIBBON_COLORS[:mat],
                hoverinfo = "skip",
                showlegend = false,
            ),
        )
        push!(
            traces,
            scatter(
                x = episodes,
                y = medians,
                mode = "lines",
                name = "median + IQR",
                legendrank = 2,
                line = attr(color = COLORS[:mat], width = 3),
                hovertemplate = "Episode %{x}<br>Median difference %{y:.2f}<extra></extra>",
            ),
        )
        push!(
            traces,
            scatter(
                x = episodes,
                y = getproperty.(aggregate, :mean),
                mode = "lines",
                name = "mean",
                legendrank = 3,
                line = attr(color = COLORS[:mat], width = 2, dash = "dot"),
                hovertemplate = "Episode %{x}<br>Mean difference %{y:.2f}<extra></extra>",
            ),
        )
        plot_object = Plot(
            traces,
            plot_layout(
                "$(uppercasefirst(string(protocol))) IC paired learning differences " *
                "(n=$(length(subset)))",
                "Episode",
                "MAT - IPPO rolling-50 score",
            ),
        )
        save_svg(
            plot_object,
            joinpath(plot_directory, "$(protocol)_paired_learning_differences"),
        )
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
    traces = PlotlyJS.GenericTrace[]
    for (position, algorithm) in enumerate((:mat, :ippo))
        values = Float64[getproperty(row, value_name) for row in rows if
                         row.algorithm == algorithm && !ismissing(getproperty(row, value_name)) &&
                         isfinite(getproperty(row, value_name))]
        isempty(values) && continue
        jitter = length(values) == 1 ? [0.0] : collect(range(-0.08, 0.08; length = length(values)))
        push!(
            traces,
            scatter(
                x = position .+ jitter,
                y = values,
                mode = "markers",
                marker = attr(
                    color = COLORS[algorithm],
                    size = 10,
                    opacity = 0.85,
                    line = attr(color = "white", width = 1),
                ),
                text = ["run $index" for index in 1:length(values)],
                hovertemplate = "%{text}<br>Value %{y:.2f}<extra>$(LABELS[algorithm])</extra>",
                showlegend = false,
            ),
        )
        push!(
            traces,
            scatter(
                x = [position - 0.16, position + 0.16],
                y = fill(median(values), 2),
                mode = "lines",
                line = attr(color = "#202020", width = 4),
                hovertemplate = "Median %{y:.2f}<extra></extra>",
                showlegend = false,
            ),
        )
        push!(
            traces,
            scatter(
                x = [position, position],
                y = [quantile(values, 0.25), quantile(values, 0.75)],
                mode = "lines",
                line = attr(color = "#202020", width = 2),
                hoverinfo = "skip",
                showlegend = false,
            ),
        )
    end
    plot_object = Plot(
        traces,
        plot_layout(
            title,
            "Algorithm",
            ylabel;
            tickvals = [1, 2],
            ticktext = ["MAT (n=$(counts[1]))", "IPPO (n=$(counts[2]))"],
            showlegend = false,
        ),
    )
    save_svg(plot_object, stem)
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
            paired_traces = PlotlyJS.GenericTrace[]
            for row in paired_complete
                push!(
                    paired_traces,
                    scatter(
                        x = [1, 2],
                        y = [row.mat_final_last100, row.ippo_final_last100],
                        mode = "lines+markers",
                        line = attr(color = "rgba(90, 90, 90, 0.45)", width = 1.5),
                        marker = attr(
                            color = [COLORS[:mat], COLORS[:ippo]],
                            size = 10,
                            line = attr(color = "white", width = 1),
                        ),
                        text = [row.run_id, row.run_id],
                        hovertemplate = "%{text}<br>Final-100 mean %{y:.2f}<extra></extra>",
                        showlegend = false,
                    ),
                )
            end
            paired_plot = Plot(
                paired_traces,
                plot_layout(
                    "$(uppercasefirst(string(protocol))) IC paired final performance " *
                    "(n=$(length(paired_complete)))",
                    "Algorithm",
                    "Mean score over last 100 training episodes";
                    tickvals = [1, 2],
                    ticktext = ["MAT", "IPPO"],
                    showlegend = false,
                ),
            )
            save_svg(
                paired_plot,
                joinpath(plot_directory, "$(protocol)_final_last100_paired"),
            )
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
        indices = collect(1:length(differences))
        paired_traces = PlotlyJS.GenericTrace[
            scatter(
                x = [first(indices), last(indices)],
                y = [0.0, 0.0],
                mode = "lines",
                line = attr(color = "#303030", width = 1.5, dash = "dash"),
                hoverinfo = "skip",
                showlegend = false,
            ),
            scatter(
                x = indices,
                y = differences,
                mode = "markers",
                marker = attr(
                    color = COLORS[:mat],
                    size = 11,
                    line = attr(color = "white", width = 1),
                ),
                hovertemplate = "Pair %{x}<br>MAT - IPPO %{y:.2f}<extra></extra>",
                showlegend = false,
            ),
            scatter(
                x = [first(indices), last(indices)],
                y = fill(median(differences), 2),
                mode = "lines",
                line = attr(color = COLORS[:mat], width = 2.5),
                hovertemplate = "Median %{y:.2f}<extra></extra>",
                showlegend = false,
            ),
        ]
        paired_plot = Plot(
            paired_traces,
            plot_layout(
                "$(uppercasefirst(string(protocol))) IC paired differences",
                "Paired seed index",
                "MAT - IPPO last-100 score";
                showlegend = false,
            ),
        )
        save_svg(paired_plot, joinpath(plot_directory, "$(protocol)_paired_differences"))
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
