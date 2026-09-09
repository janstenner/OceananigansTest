ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using JLD2
using PlotlyJS
using Statistics

# None of the paper-figure labels require MathJax. Disabling it avoids a
# spurious missing-extension message in Kaleido's Windows PDF output.
PlotlyJS.PlotlyKaleido.kill_kaleido()
PlotlyJS.PlotlyKaleido.start(plotlyjs = PlotlyJS._js_path, mathjax = false)

const WINDOW = 50
const PROTOCOLS = (:fixed, :varying)
const SERIES = (:dense, :gc, :sc)
const EPISODE_TARGETS = Dict(:fixed => 2_000, :varying => 4_000)
const SERIES_LABELS = Dict(
    :dense => "MAT (full sensor set)",
    :gc => "MAT (GC mask)",
    :sc => "MAT (SC mask)",
)
# The three configuration colors from MAT_Stability/collect_results.jl.
const SERIES_COLORS = Dict(
    :dense => "#277DA1",
    :gc => "#F2A13A",
    :sc => "#B41A5C",
)
const SERIES_RUN_COLORS = Dict(
    :dense => "rgba(39, 125, 161, 0.22)",
    :gc => "rgba(242, 161, 58, 0.22)",
    :sc => "rgba(180, 35, 97, 0.22)",
)
const SERIES_RIBBON_COLORS = Dict(
    :dense => "rgba(39, 125, 161, 0.18)",
    :gc => "rgba(242, 161, 58, 0.18)",
    :sc => "rgba(180, 35, 97, 0.18)",
)
const DEFAULT_RESULTS = joinpath(@__DIR__, "results")
const DEFAULT_COMPARISON_RESULTS = joinpath(@__DIR__, "..", "MAT_IPPO_Comparison", "results")

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/MaskedTraining/make_paper_figures.jl [options]

    Options:
      --results-dir PATH             MaskedTraining result root (default: results).
      --comparison-results-dir PATH  MAT_IPPO_Comparison result root.
      --output-dir PATH              Output directory (default: <results>/paper).
      --check-only                   Validate all paired inputs without rendering.
      --help

    Produces learning_curves_combined.svg/pdf and final_100_statistics.csv.
    The two-panel figure compares the ten original MAT runs with the paired
    GC- and SC-masked runs under Fixed and Varying IC.
    """)
end

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "results_dir" => DEFAULT_RESULTS,
        "comparison_results_dir" => DEFAULT_COMPARISON_RESULTS,
        "output_dir" => nothing,
        "check_only" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--check-only"
            values["check_only"] = true
            index += 1
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
    results = abspath(string(values["results_dir"]))
    comparison = abspath(string(values["comparison_results_dir"]))
    output = isnothing(values["output_dir"]) ? joinpath(results, "paper") : abspath(string(values["output_dir"]))
    return (; results, comparison, output, check_only = Bool(values["check_only"]))
end

optional_read(file, key, default) = haskey(file, key) ? read(file, key) : default

function read_complete_rewards(path, expected; manifest_identity = nothing)
    isfile(path) || error("Training result is missing: $path")
    return JLD2.jldopen(path, "r") do file
        string(read(file, "status")) == "complete" || error("Training result is not complete: $path")
        rewards = Float64.(read(file, "rewards"))
        Int(read(file, "episode_target")) == expected || error("Episode target mismatch: $path")
        Int(read(file, "episodes_completed")) == expected || error("Episode count mismatch: $path")
        length(rewards) == expected || error("Reward count mismatch: $path")
        all(isfinite, rewards) || error("Non-finite reward in $path")
        if !isnothing(manifest_identity)
            string(read(file, "manifest_identity")) == manifest_identity || error("Manifest identity mismatch: $path")
        end
        return (
            rewards = rewards,
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
        )
    end
end

function load_curves(results, comparison)
    manifest_path = joinpath(results, "manifest.jld2")
    isfile(manifest_path) || error("MaskedTraining manifest is missing: $manifest_path")
    manifest = JLD2.load(manifest_path, "manifest")
    length(manifest.entries) == 10 || error("Expected ten manifest entries, found $(length(manifest.entries)).")
    length(unique(entry.run_id for entry in manifest.entries)) == 10 || error("Manifest run IDs are not unique.")
    curves = Dict{Tuple{Symbol, Symbol}, Vector{Vector{Float64}}}()
    for protocol in PROTOCOLS
        expected = EPISODE_TARGETS[protocol]
        per_series = Dict(series => Vector{Vector{Float64}}() for series in SERIES)
        for entry in manifest.entries
            dense_path = joinpath(comparison, "runs", entry.run_id, string(protocol), "mat.jld2")
            dense = read_complete_rewards(dense_path, expected)
            dense.run_seed == entry.run_seed && dense.ic_seed == entry.ic_seed || error("Dense seed mismatch: $dense_path")
            push!(per_series[:dense], dense.rewards)
            for grouping in (:gc, :sc)
                masked_path = joinpath(results, "runs", string(protocol), string(grouping), "$(entry.run_id).jld2")
                masked = read_complete_rewards(masked_path, expected; manifest_identity = string(manifest.identity))
                masked.run_seed == entry.run_seed && masked.ic_seed == entry.ic_seed || error("Masked seed mismatch: $masked_path")
                push!(per_series[grouping], masked.rewards)
            end
        end
        for series in SERIES
            length(per_series[series]) == 10 || error("Expected ten $protocol/$series curves.")
            curves[(protocol, series)] = per_series[series]
        end
    end
    return curves
end

rolling_mean(values, width = WINDOW) = [
    mean(@view values[(index - width + 1):index]) for index in width:length(values)
]

function aggregate_curves(raw_curves)
    rolled = rolling_mean.(raw_curves)
    count = minimum(length, rolled)
    values = reduce(hcat, (curve[1:count] for curve in rolled))
    return (
        runs = rolled,
        episodes = collect(WINDOW:(WINDOW + count - 1)),
        mean = vec(mean(values; dims = 2)),
        median = [median(@view values[index, :]) for index in 1:count],
        q25 = [quantile(@view(values[index, :]), 0.25) for index in 1:count],
        q75 = [quantile(@view(values[index, :]), 0.75) for index in 1:count],
    )
end

function learning_curves_combined(curves)
    plot_handle = make_subplots(
        rows = 1,
        cols = 2,
        horizontal_spacing = 0.08,
        subplot_titles = reshape(["(a) Fixed IC", "(b) Varying IC"], :, 1),
    )
    for (column, protocol) in enumerate(PROTOCOLS), (series_index, series) in enumerate(SERIES)
        aggregate = aggregate_curves(curves[(protocol, series)])
        add_trace!(plot_handle, scatter(
            x = aggregate.episodes, y = aggregate.q25, mode = "lines",
            line = attr(width = 0), hoverinfo = "skip", showlegend = false,
        ); row = 1, col = column)
        add_trace!(plot_handle, scatter(
            x = aggregate.episodes, y = aggregate.q75, mode = "lines",
            line = attr(width = 0), fill = "tonexty",
            fillcolor = SERIES_RIBBON_COLORS[series], hoverinfo = "skip", showlegend = false,
        ); row = 1, col = column)
        for run in aggregate.runs
            add_trace!(plot_handle, scatter(
                x = aggregate.episodes, y = run, mode = "lines",
                line = attr(color = SERIES_RUN_COLORS[series], width = 1.25),
                hoverinfo = "skip", showlegend = false,
            ); row = 1, col = column)
        end
        add_trace!(plot_handle, scatter(
            x = aggregate.episodes, y = aggregate.median, mode = "lines",
            name = SERIES_LABELS[series], legendrank = series_index,
            showlegend = column == 1,
            line = attr(color = SERIES_COLORS[series], width = 3),
            hovertemplate = "Episode %{x}<br>Median %{y:.2f}<extra>%{fullData.name}</extra>",
        ); row = 1, col = column)
        add_trace!(plot_handle, scatter(
            x = aggregate.episodes, y = aggregate.mean, mode = "lines",
            line = attr(color = SERIES_COLORS[series], width = 2, dash = "dash"),
            hovertemplate = "Episode %{x}<br>Mean %{y:.2f}<extra>$(SERIES_LABELS[series])</extra>",
            showlegend = false,
        ); row = 1, col = column)
    end
    for (name, rank, line_style) in (
        ("Individual runs", 10, attr(color = "rgba(70, 70, 70, 0.30)", width = 1.25)),
        ("Median + IQR", 11, attr(color = "#555555", width = 3)),
        ("Arithmetic mean", 12, attr(color = "#555555", width = 2, dash = "dash")),
    )
        add_trace!(plot_handle, scatter(x = [NaN], y = [NaN], mode = "lines", name = name,
            legendrank = rank, line = line_style, hoverinfo = "skip"); row = 1, col = 1)
    end
    xaxis_style = attr(
        title = attr(text = "Episode", standoff = 12), showline = true,
        mirror = true, linecolor = "#3A3A3A", linewidth = 1, ticks = "outside",
        gridcolor = "#E6E6E6", zeroline = false,
    )
    yaxis_style = attr(
        showline = true, mirror = true, linecolor = "#3A3A3A", linewidth = 1,
        ticks = "outside", gridcolor = "#E6E6E6", zeroline = false,
    )
    yaxis_fields = Dict{Symbol, Any}(yaxis_style.fields)
    yaxis_fields[:title] = attr(text = "Score (rolling mean, window=$WINDOW)", standoff = 12)
    annotations = plot_handle.plot.layout.fields[:annotations]
    annotations[1].fields[:x] = 0.23
    annotations[2].fields[:x] = 0.77
    for annotation in annotations
        annotation.fields[:font] = attr(size = 30, color = "#252525")
    end
    relayout!(plot_handle,
        template = "plotly_white", paper_bgcolor = "white", plot_bgcolor = "white",
        width = 1500, height = 700,
        margin = attr(l = 125, r = 35, t = 85, b = 175),
        font = attr(family = "Arial, sans-serif", size = 26, color = "#303030"),
        xaxis = xaxis_style, xaxis2 = xaxis_style,
        yaxis = attr(; yaxis_fields...), yaxis2 = yaxis_style,
        legend = attr(orientation = "h", x = 0.5, y = -0.24, xanchor = "center",
            yanchor = "top", traceorder = "normal", bgcolor = "rgba(255, 255, 255, 0.92)",
            bordercolor = "#CFCFCF", borderwidth = 1, font = attr(size = 24)),
        hovermode = "x unified",
    )
    return plot_handle
end

function final_100_statistics(curves)
    rows = NamedTuple[]
    for protocol in PROTOCOLS
        dense = mean.(last.(curves[(protocol, :dense)], Ref(100)))
        for series in SERIES
            values = mean.(last.(curves[(protocol, series)], Ref(100)))
            differences = series === :dense ? nothing : values .- dense
            aggregate = aggregate_curves(curves[(protocol, series)])
            range = findall(>=(500), aggregate.episodes)
            push!(rows, (
                protocol = protocol, series = series, n = length(values),
                mean = mean(values), sample_sd = std(values), median = median(values),
                mean_paired_difference_from_dense = isnothing(differences) ? missing : mean(differences),
                sample_sd_paired_difference = isnothing(differences) ? missing : std(differences),
                median_paired_difference_from_dense = isnothing(differences) ? missing : median(differences),
                paired_wins = isnothing(differences) ? missing : count(>(0), differences),
                mean_pointwise_iqr_episodes_500_to_end = mean((aggregate.q75 .- aggregate.q25)[range]),
            ))
        end
    end
    return rows
end

function write_statistics_csv(path, rows)
    columns = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(columns, ','))
        for row in rows
            println(io, join((ismissing(getproperty(row, key)) ? "" : getproperty(row, key) for key in columns), ','))
        end
    end
    return path
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    curves = load_curves(options.results, options.comparison)
    println("Validated ten paired Dense/GC/SC runs for Fixed and Varying IC.")
    options.check_only && return String[]
    mkpath(options.output)
    plot_handle = learning_curves_combined(curves)
    outputs = [joinpath(options.output, "learning_curves_combined.$extension") for extension in ("svg", "pdf")]
    for output in outputs
        PlotlyJS.savefig(plot_handle, output; width = 1500, height = 700)
        println("Wrote $output")
    end
    statistics_path = write_statistics_csv(joinpath(options.output, "final_100_statistics.csv"), final_100_statistics(curves))
    push!(outputs, statistics_path)
    println("Wrote $statistics_path")
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
