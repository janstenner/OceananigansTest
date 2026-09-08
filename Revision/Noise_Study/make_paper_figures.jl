ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Dates
using JLD2
using PlotlyJS
using Printf
using SHA
using Statistics

PlotlyJS.PlotlyKaleido.kill_kaleido()
PlotlyJS.PlotlyKaleido.start(plotlyjs = PlotlyJS._js_path, mathjax = false)

include(joinpath(@__DIR__, "NoiseStudy.jl"))
using .NoiseStudy

const DEFAULT_NOISE_RESULTS = joinpath(@__DIR__, "results")
const PAPER_PROTOCOLS = (:fixed, :varying)
const PAPER_CONTROLLERS = (:expert, :c_match, :sparse)
const PAPER_PLOT_CONTROLLERS = (:expert, :sparse, :c_match)
const PAPER_CONTROLLER_LABELS = Dict(
    :expert => "Dense expert",
    :sparse => "Sparse apprentice",
    :c_match => "C_match apprentice",
)
const PAPER_PLOT_LABELS = Dict(
    :expert => "Dense expert",
    :sparse => "Sparse apprentice",
    :c_match => "C_match",
)
# Reuse the three principal colors from the MAT-stability figures in the
# requested expert, sparse-apprentice, C_match order.
const PAPER_CONTROLLER_COLORS = Dict(
    :expert => "#277DA1",
    :sparse => "#F2A13A",
    :c_match => "#B41A5C",
)
const PAPER_CONTROLLER_SYMBOLS = Dict(
    :expert => "circle",
    :sparse => "diamond",
    :c_match => "square",
)
const PAPER_PROTOCOL_LABELS = Dict(:fixed => "Fixed IC", :varying => "Varying IC")
const PAPER_FIGURE_STEMS = Dict(
    :fixed => "figure_1a_noise_robustness_fixed",
    :varying => "figure_1b_noise_robustness_varying",
)
const PAPER_CHANNEL_COLORS = ("#277DA1", "#F2A13A", "#B41A5C")
const PAPER_CHANNEL_NAMES = ("Temperature", "Vertical velocity", "Horizontal velocity")
const PAPER_INACTIVE_COLOR = "#F2F2F2"
const PAPER_FONT_SIZE = 22
const PAPER_AXIS_TITLE_SIZE = 22
const PAPER_TICK_SIZE = 18
const PAPER_TITLE_SIZE = 30
const PAPER_SUBPLOT_TITLE_SIZE = 22
const PAPER_LEGEND_SIZE = 18
const STRIPE_CHANNEL_WIDTH = 4
const STRIPE_SENSOR_WIDTH = 3 * STRIPE_CHANNEL_WIDTH + 1
const STRIPE_COLUMN_COUNT = 48 * STRIPE_SENSOR_WIDTH - 1

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/Noise_Study/make_paper_figures.jl \\
        [--experiment-id ID] [--results-dir PATH] [--output-dir PATH] [--check-only]

    Without --experiment-id, the script independently selects the newest
    Noise-Study experiment directory available for Fixed and Varying IC.
    Different or relocated experiment IDs produce warnings but do not abort.
    Missing worker results are written as NA so partial tables and figures can
    be created while the other protocol is still running. Each available
    protocol produces a separate paper-ready SVG and PDF noise-response plot,
    and the two frozen C_match masks are combined in one supplementary figure.
    """)
end

function validated_experiment_id(value)
    identifier = strip(string(value))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", identifier) || error(
        "Invalid experiment ID '$identifier'.",
    )
    return identifier
end

function parse_arguments(arguments)
    options = Dict{String, Any}(
        "experiment_id" => nothing,
        "results_dir" => DEFAULT_NOISE_RESULTS,
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
            options["check_only"] = true
            index += 1
        elseif startswith(argument, "--")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(options, key) || error("Unknown option '$argument'.")
            options[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument '$argument'.")
        end
    end
    results_root = abspath(string(options["results_dir"]))
    isdir(results_root) || error("Noise-Study results directory is missing: $results_root")
    common_experiment = isnothing(options["experiment_id"]) ? nothing :
        validated_experiment_id(options["experiment_id"])
    experiments = select_experiments(results_root, common_experiment)
    isempty(experiments) && error("No Fixed or Varying Noise-Study experiment directories found.")
    output = if !isnothing(options["output_dir"])
        abspath(string(options["output_dir"]))
    elseif length(unique(values(experiments))) == 1
        joinpath(results_root, only(unique(values(experiments))), "paper")
    else
        fixed_tag = get(experiments, :fixed, "missing_fixed")
        varying_tag = get(experiments, :varying, "missing_varying")
        joinpath(results_root, "paper_tables", "fixed_$(fixed_tag)__varying_$(varying_tag)")
    end
    return (;
        results_root,
        experiments,
        output,
        check_only = Bool(options["check_only"]),
    )
end

function protocol_experiment_directories(results_root, protocol::Symbol)
    directories = String[]
    for name in readdir(results_root)
        startswith(name, ".") && continue
        root = joinpath(results_root, name)
        isdir(root) || continue
        try
            validated_experiment_id(name)
        catch
            continue
        end
        protocol_root = joinpath(root, string(protocol))
        manifest = joinpath(root, "manifests", "$(protocol).jld2")
        if isdir(protocol_root) || isfile(manifest)
            push!(directories, root)
        end
    end
    return directories
end

function newest_experiment_directory(directories)
    timestamped = filter(path -> occursin(r"^\d{6}_\d{6}$", basename(path)), directories)
    return isempty(timestamped) ?
        last(sort(directories; by = path -> (stat(path).mtime, basename(path)))) :
        last(sort(timestamped; by = basename))
end

function latest_protocol_experiment(results_root, protocol::Symbol)
    directories = protocol_experiment_directories(results_root, protocol)
    isempty(directories) && return nothing
    identifiers = sort!(validated_experiment_id.(basename.(directories)))
    if length(identifiers) > 1
        @warn "Multiple Noise-Study experiment IDs found; continuing with the newest one." protocol available_experiment_ids=identifiers
    end
    selected = newest_experiment_directory(directories)
    identifier = validated_experiment_id(basename(selected))
    println("No --experiment-id supplied; using newest $(protocol) Noise-Study directory: $identifier")
    return identifier
end

function select_experiments(results_root, common_experiment)
    selected = Dict{Symbol, String}()
    for protocol in PAPER_PROTOCOLS
        identifier = if isnothing(common_experiment)
            latest_protocol_experiment(results_root, protocol)
        else
            protocol_root = joinpath(results_root, common_experiment, string(protocol))
            manifest = joinpath(results_root, common_experiment, "manifests", "$(protocol).jld2")
            if !isdir(protocol_root) && !isfile(manifest)
                @warn "Requested experiment has no artifacts for this protocol; continuing without it." protocol experiment_id=common_experiment
                nothing
            else
                common_experiment
            end
        end
        isnothing(identifier) || (selected[protocol] = identifier)
    end
    selected_ids = unique(values(selected))
    if length(selected_ids) > 1
        @warn "Fixed and Varying Noise-Study tables use different experiment IDs; continuing." fixed_experiment_id=get(selected, :fixed, nothing) varying_experiment_id=get(selected, :varying, nothing)
    end
    return selected
end

valueof(value, key::Symbol) = value isa NamedTuple ? getproperty(value, key) :
    value isa AbstractDict ? (haskey(value, key) ? value[key] : value[string(key)]) :
    getproperty(value, key)

function load_controller_metadata(results_root, experiment_id, protocol::Symbol)
    path = joinpath(results_root, experiment_id, "manifests", "$(protocol).jld2")
    if !isfile(path)
        @warn "Noise-Study manifest is missing; using result metadata and default labels." protocol experiment_id path
        return (path = nothing, controllers = Dict{Symbol, Any}())
    end
    manifest = JLD2.load(path)
    stored_experiment_id = string(manifest["experiment_id"])
    if stored_experiment_id != experiment_id
        @warn "Manifest experiment mismatch; continuing with relocated artifacts." protocol selected_experiment_id=experiment_id stored_experiment_id manifest_path=path
    end
    Symbol(manifest["protocol"]) === protocol || error("Manifest protocol mismatch: $path")
    controllers = Dict{Symbol, Any}(
        Symbol(valueof(record, :kind)) => record for record in manifest["controllers"]
    )
    return (path = abspath(path), controllers)
end

function result_file(results_root, experiment_id, protocol, controller, level)
    return joinpath(
        results_root,
        experiment_id,
        string(protocol),
        string(controller),
        level_tag(level),
        "result.jld2",
    )
end

function load_worker_result(path, experiment_id, protocol, controller, level)
    loaded = JLD2.load(path)
    Symbol(loaded["experiment"]) === :package10_sensor_noise_worker || error(
        "Unexpected worker experiment in $path",
    )
    stored_experiment_id = string(loaded["experiment_id"])
    if stored_experiment_id != experiment_id
        @warn "Worker experiment mismatch; continuing with relocated artifacts." protocol controller noise_level=level selected_experiment_id=experiment_id stored_experiment_id result_path=path
    end
    Symbol(loaded["protocol"]) === protocol || error("Worker protocol mismatch: $path")
    Symbol(loaded["controller"]) === controller || error("Worker controller mismatch: $path")
    Float64(loaded["noise_level"]) == Float64(level) || error("Worker noise-level mismatch: $path")
    summaries = loaded["summaries"]
    case_count = Int(loaded["case_count"])
    replicate_count = Int(loaded["replicate_count"])
    expected_episodes = level == 0 ? case_count : case_count * replicate_count
    length(summaries) == expected_episodes || error(
        "Worker summary count mismatch in $path: expected $expected_episodes, found $(length(summaries)).",
    )
    mean_state_nusselt = mean(Float64(valueof(summary, :mean_state_nusselt)) for summary in summaries)
    return (;
        path = abspath(path),
        stored_experiment_id,
        configuration = string(loaded["configuration"]),
        controller_id = string(loaded["controller_id"]),
        case_count,
        replicate_count,
        episode_count = length(summaries),
        mean_state_nusselt,
    )
end

function controller_configuration(metadata, controller::Symbol, available_results)
    if haskey(metadata, controller)
        return string(valueof(metadata[controller], :configuration))
    end
    for result in available_results
        isnothing(result) || return result.configuration
    end
    return controller === :expert ? "dense" : controller === :c_match ? "go-sc" : "unknown"
end

function load_protocol_table(options, protocol::Symbol, experiment_id::String)
    metadata = load_controller_metadata(options.results_root, experiment_id, protocol)
    rows = NamedTuple[]
    long_rows = NamedTuple[]
    source_files = String[]
    isnothing(metadata.path) || push!(source_files, metadata.path)
    missing_results = String[]
    for controller in PAPER_CONTROLLERS
        results = Any[]
        for level in NOISE_LEVELS
            path = result_file(options.results_root, experiment_id, protocol, controller, level)
            if isfile(path)
                result = load_worker_result(path, experiment_id, protocol, controller, level)
                push!(results, result)
                push!(source_files, result.path)
                push!(long_rows, (
                    protocol,
                    experiment_id,
                    controller,
                    candidate = PAPER_CONTROLLER_LABELS[controller],
                    configuration = result.configuration,
                    noise_level = Float64(level),
                    mean_test_state_nu = result.mean_state_nusselt,
                    case_count = result.case_count,
                    replicate_count = result.replicate_count,
                    episode_count = result.episode_count,
                    result_path = result.path,
                ))
            else
                push!(results, nothing)
                push!(missing_results, replace(relpath(path, options.results_root), '\\' => '/'))
            end
        end
        configuration = controller_configuration(metadata.controllers, controller, results)
        values_by_level = Dict(
            Float64(level) => (isnothing(result) ? missing : result.mean_state_nusselt)
            for (level, result) in zip(NOISE_LEVELS, results)
        )
        push!(rows, (;
            protocol,
            experiment_id,
            controller,
            candidate = PAPER_CONTROLLER_LABELS[controller],
            configuration,
            values_by_level,
        ))
    end
    isempty(missing_results) || @warn(
        "Noise-Study worker results are incomplete; missing table cells will be written as NA.",
        protocol,
        experiment_id,
        missing_results,
    )
    return (; rows, long_rows, source_files, missing_results)
end

function csv_escape(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

format_metric(value) = ismissing(value) ? "NA" : @sprintf("%.6f", Float64(value))

function level_column(level)
    return "noise_" * replace(@sprintf("%.2f", Float64(level)), "." => "p")
end

function write_csv(output, rows)
    path = joinpath(output, "table_1_noise_robustness.csv")
    open(path, "w") do io
        headers = vcat(
            ["protocol", "experiment_id", "candidate", "configuration"],
            level_column.(NOISE_LEVELS),
        )
        println(io, join(headers, ','))
        for row in rows
            fields = String[
                string(row.protocol),
                row.experiment_id,
                row.candidate,
                row.configuration,
            ]
            append!(fields, format_metric(row.values_by_level[Float64(level)]) for level in NOISE_LEVELS)
            println(io, join(csv_escape.(fields), ','))
        end
    end
    return abspath(path)
end

function write_markdown(output, rows, experiments)
    path = joinpath(output, "table_1_noise_robustness.md")
    open(path, "w") do io
        println(io, "# Package-10 sensor-noise robustness")
        println(io)
        headers = vcat(
            ["Protocol", "Candidate", "Configuration"],
            [@sprintf("Noise %.2f", Float64(level)) for level in NOISE_LEVELS],
        )
        println(io, "| " * join(headers, " | ") * " |")
        println(io, "|" * join(fill("---", length(headers)), "|") * "|")
        for row in rows
            fields = String[
                row.protocol === :fixed ? "Fixed IC" : "Varying IC",
                row.candidate,
                row.configuration,
            ]
            append!(fields, format_metric(row.values_by_level[Float64(level)]) for level in NOISE_LEVELS)
            println(io, "| " * join(fields, " | ") * " |")
        end
        println(io)
        println(io, "Each value is the mean of the stored per-episode `mean_state_nusselt` values over the complete available test set and all noise replicates. Lower is better. `NA` denotes a worker result that was not complete when the table was created.")
        println(io)
        for protocol in PAPER_PROTOCOLS
            haskey(experiments, protocol) || continue
            println(io, "- $(protocol): experiment `$(experiments[protocol])`")
        end
    end
    return abspath(path)
end

function stripe_matrix(mask)
    size(mask) == (3, 48, 8) || error("Expected a 3×48×8 global mask, found $(size(mask)).")
    values = fill(NaN, 8, STRIPE_COLUMN_COUNT)
    text = fill("", 8, STRIPE_COLUMN_COUNT)
    for vertical in 1:8, horizontal in 1:48, channel in 1:3
        active = Bool(mask[channel, horizontal, vertical])
        start = (horizontal - 1) * STRIPE_SENSOR_WIDTH + (channel - 1) * STRIPE_CHANNEL_WIDTH + 1
        for column in start:(start + STRIPE_CHANNEL_WIDTH - 1)
            values[vertical, column] = active ? channel : 0
            text[vertical, column] = "x=$horizontal, z=$vertical<br>$(PAPER_CHANNEL_NAMES[channel]): $(active ? "active" : "inactive")"
        end
    end
    return values, text
end

stripe_sensor_center(horizontal) =
    (horizontal - 1) * STRIPE_SENSOR_WIDTH + (3 * STRIPE_CHANNEL_WIDTH + 1) / 2

const STRIPE_COLORSCALE = [
    [0.0, PAPER_INACTIVE_COLOR], [1 / 6, PAPER_INACTIVE_COLOR],
    [1 / 6, PAPER_CHANNEL_COLORS[1]], [0.5, PAPER_CHANNEL_COLORS[1]],
    [0.5, PAPER_CHANNEL_COLORS[2]], [5 / 6, PAPER_CHANNEL_COLORS[2]],
    [5 / 6, PAPER_CHANNEL_COLORS[3]], [1.0, PAPER_CHANNEL_COLORS[3]],
]

function preserved_axis(plot, key, styling)
    existing = get(plot.plot.layout.fields, key, Dict{Any, Any}())
    fields = Dict{Symbol, Any}(Symbol(name) => value for (name, value) in existing)
    merge!(fields, styling.fields)
    return PlotlyJS.attr(; fields...)
end

axis_key(axis, index) = Symbol(index == 1 ? axis : "$(axis)$(index)")

function paper_axis(title; kwargs...)
    fields = Dict{Symbol, Any}(
        :title => PlotlyJS.attr(text = title, standoff = 12, font = PlotlyJS.attr(size = PAPER_AXIS_TITLE_SIZE)),
        :tickfont => PlotlyJS.attr(size = PAPER_TICK_SIZE),
    )
    for (key, value) in kwargs
        fields[key] = value
    end
    return PlotlyJS.attr(; fields...)
end

function load_c_match_masks()
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    go_study_results = source_defaults(project_root).go_study_results
    candidates = Dict(
        protocol => load_c_match_candidate(protocol, go_study_results)
        for protocol in PAPER_PROTOCOLS
    )
    for protocol in PAPER_PROTOCOLS
        candidate = candidates[protocol].candidate
        Symbol(candidate[:selection_role]) === :C_match || error("Unexpected selection role for $protocol C_match mask.")
        size(candidate[:global_mask]) == (3, 48, 8) || error("Unexpected global-mask shape for $protocol C_match.")
    end
    return candidates
end

function write_c_match_mask_plot(output, candidates)
    plot = PlotlyJS.make_subplots(
        rows = 1,
        cols = 2,
        horizontal_spacing = 0.07,
        subplot_titles = reshape(["Fixed IC", "Varying IC"], :, 1),
    )
    annotations = get(plot.plot.layout.fields, :annotations, Any[])
    for annotation in annotations
        annotation.fields[:font] = PlotlyJS.attr(size = PAPER_SUBPLOT_TITLE_SIZE, color = "#252525")
    end
    for (column, protocol) in enumerate(PAPER_PROTOCOLS)
        values, text = stripe_matrix(candidates[protocol].candidate[:global_mask])
        PlotlyJS.add_trace!(plot, PlotlyJS.heatmap(
            x = collect(1:STRIPE_COLUMN_COUNT),
            y = collect(1:8),
            z = values,
            text = text,
            zmin = 0,
            zmax = 3,
            colorscale = STRIPE_COLORSCALE,
            showscale = false,
            hovertemplate = "%{text}<extra></extra>",
            xgap = 0,
            ygap = 1,
        ); row = 1, col = column)
    end
    for (index, (name, color)) in enumerate(zip(
        ("Inactive", PAPER_CHANNEL_NAMES...),
        (PAPER_INACTIVE_COLOR, PAPER_CHANNEL_COLORS...),
    ))
        PlotlyJS.add_trace!(plot, PlotlyJS.scatter(
            x = [NaN],
            y = [NaN],
            mode = "markers",
            name = name,
            marker = PlotlyJS.attr(
                color = color,
                size = 11,
                symbol = "square",
                line = PlotlyJS.attr(color = "#444444", width = index == 1 ? 1 : 0),
            ),
            showlegend = true,
        ); row = 1, col = 1)
    end
    layout = Dict{Symbol, Any}(
        :template => "plotly_white",
        :width => 1450,
        :height => 650,
        :title => PlotlyJS.attr(
            text = "Noise robustness: selected C_match sensor masks",
            x = 0.5,
            xanchor = "center",
            font = PlotlyJS.attr(size = PAPER_TITLE_SIZE, color = "#252525"),
        ),
        :paper_bgcolor => "white",
        :plot_bgcolor => "white",
        :font => PlotlyJS.attr(family = "Arial, sans-serif", size = PAPER_FONT_SIZE, color = "#303030"),
        :margin => PlotlyJS.attr(l = 105, r = 35, t = 120, b = 130),
        :legend => PlotlyJS.attr(
            orientation = "h",
            x = 0.5,
            xanchor = "center",
            y = -0.20,
            yanchor = "top",
            font = PlotlyJS.attr(size = PAPER_LEGEND_SIZE),
        ),
    )
    for index in 1:2
        layout[axis_key("xaxis", index)] = preserved_axis(plot, axis_key("xaxis", index), paper_axis(
            "Horizontal sensor index",
            range = [0.5, STRIPE_COLUMN_COUNT + 0.5],
            tickmode = "array",
            tickvals = [stripe_sensor_center(value) for value in 1:4:48],
            ticktext = string.(1:4:48),
            showgrid = false,
            zeroline = false,
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
        ))
        layout[axis_key("yaxis", index)] = preserved_axis(plot, axis_key("yaxis", index), paper_axis(
            index == 1 ? "Vertical sensor index" : "",
            range = [0.5, 8.5],
            tickmode = "array",
            tickvals = collect(1:8),
            showgrid = false,
            zeroline = false,
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
        ))
    end
    PlotlyJS.relayout!(plot, layout)
    stem = "figure_s1_c_match_sensor_masks"
    svg_path = abspath(joinpath(output, "$stem.svg"))
    pdf_path = abspath(joinpath(output, "$stem.pdf"))
    PlotlyJS.savefig(plot, svg_path; width = 1450, height = 650)
    PlotlyJS.savefig(plot, pdf_path; width = 1450, height = 650)
    keep_first_pdf_page!(pdf_path)
    return (; svg_path, pdf_path)
end

function keep_first_pdf_page!(path)
    pdfseparate = Sys.which("pdfseparate")
    if isnothing(pdfseparate)
        @warn "pdfseparate is unavailable; leaving the raw Kaleido PDF unchanged." path
        return path
    end
    mktempdir() do temporary
        pattern = joinpath(temporary, "page-%d.pdf")
        run(`$pdfseparate -f 1 -l 1 $path $pattern`)
        first_page = joinpath(temporary, "page-1.pdf")
        isfile(first_page) || error("pdfseparate did not create the expected first page for $path")
        mv(first_page, path; force = true)
    end
    return path
end

function write_protocol_plot(output, rows, protocol::Symbol)
    protocol_rows = filter(row -> row.protocol === protocol, rows)
    isempty(protocol_rows) && return nothing
    traces = PlotlyJS.GenericTrace[]
    for controller in PAPER_PLOT_CONTROLLERS
        selected = filter(row -> row.controller === controller, protocol_rows)
        isempty(selected) && continue
        length(selected) == 1 || error("Expected one $protocol/$controller paper row, found $(length(selected)).")
        row = only(selected)
        values = [row.values_by_level[Float64(level)] for level in NOISE_LEVELS]
        y_values = [ismissing(value) ? NaN : Float64(value) for value in values]
        all(isnan, y_values) && continue
        push!(traces, PlotlyJS.scatter(
            x = Float64.(collect(NOISE_LEVELS)),
            y = y_values,
            customdata = Float64.(collect(NOISE_LEVELS)),
            mode = "lines+markers",
            name = PAPER_PLOT_LABELS[controller],
            connectgaps = false,
            line = PlotlyJS.attr(color = PAPER_CONTROLLER_COLORS[controller], width = 3),
            marker = PlotlyJS.attr(
                color = PAPER_CONTROLLER_COLORS[controller],
                symbol = PAPER_CONTROLLER_SYMBOLS[controller],
                size = 10,
                line = PlotlyJS.attr(color = "white", width = 1),
            ),
            hovertemplate = "Noise level α=%{customdata:.2f}<br>Mean test-set Nu=%{y:.5f}<extra>$(PAPER_PLOT_LABELS[controller])</extra>",
        ))
    end
    if isempty(traces)
        @warn "No complete Noise-Study points are available for a paper plot." protocol
        return nothing
    end
    figure = PlotlyJS.Plot(traces, PlotlyJS.Layout(
        template = "plotly_white",
        width = 950,
        height = 620,
        title = PlotlyJS.attr(
            text = "$(PAPER_PROTOCOL_LABELS[protocol]) sensor-noise robustness",
            x = 0.5,
            xanchor = "center",
            font = PlotlyJS.attr(size = 28, color = "#252525"),
        ),
        font = PlotlyJS.attr(family = "Arial, sans-serif", size = 22, color = "#303030"),
        xaxis = PlotlyJS.attr(
            title = PlotlyJS.attr(text = "Evaluated relative noise level α", standoff = 12, font = PlotlyJS.attr(size = 22)),
            tickmode = "linear",
            tick0 = 0.0,
            dtick = 0.1,
            tickformat = ".1f",
            tickfont = PlotlyJS.attr(size = 18),
            range = [-0.03, 1.03],
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        yaxis = PlotlyJS.attr(
            title = PlotlyJS.attr(text = "Mean test-set Nusselt number", standoff = 12, font = PlotlyJS.attr(size = 22)),
            tickfont = PlotlyJS.attr(size = 18),
            tickformat = ".2f",
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        legend = PlotlyJS.attr(
            orientation = "h",
            x = 0.5,
            y = -0.24,
            xanchor = "center",
            yanchor = "top",
            traceorder = "normal",
            bgcolor = "rgba(255, 255, 255, 0.92)",
            bordercolor = "#CFCFCF",
            borderwidth = 1,
            font = PlotlyJS.attr(size = 18),
        ),
        hovermode = "x unified",
        margin = PlotlyJS.attr(l = 110, r = 35, t = 90, b = 145),
    ))
    stem = PAPER_FIGURE_STEMS[protocol]
    svg_path = abspath(joinpath(output, "$stem.svg"))
    pdf_path = abspath(joinpath(output, "$stem.pdf"))
    PlotlyJS.savefig(figure, svg_path; width = 950, height = 620)
    PlotlyJS.savefig(figure, pdf_path; width = 950, height = 620)
    keep_first_pdf_page!(pdf_path)
    return (; svg_path, pdf_path)
end

file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function write_provenance(output, source_files, experiments)
    path = joinpath(output, "provenance.sha256")
    files = sort!(unique!(abspath.(source_files)))
    open(path, "w") do io
        println(io, "# Package-10 paper figure and table input manifest")
        println(io, "# Script SHA-256: $(file_sha256(abspath(@__FILE__)))")
        println(io, "# Experiment IDs: $(join(("$(protocol)=$(identifier)" for (protocol, identifier) in sort!(collect(experiments); by = pair -> string(first(pair)))), ", "))")
        for file in files
            println(io, "$(file_sha256(file))  $(replace(relpath(file, output), '\\' => '/'))")
        end
    end
    return abspath(path)
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    rows = NamedTuple[]
    long_rows = NamedTuple[]
    source_files = String[]
    missing_results = String[]
    for protocol in PAPER_PROTOCOLS
        haskey(options.experiments, protocol) || continue
        loaded = load_protocol_table(options, protocol, options.experiments[protocol])
        append!(rows, loaded.rows)
        append!(long_rows, loaded.long_rows)
        append!(source_files, loaded.source_files)
        append!(missing_results, loaded.missing_results)
    end
    isempty(rows) && error("No Noise-Study table rows could be assembled.")
    c_match_masks = load_c_match_masks()
    for selected in values(c_match_masks)
        push!(source_files, selected.selection_path)
        push!(source_files, selected.checkpoint_path)
    end
    if options.check_only
        println("Noise-Study paper inputs read: $(length(long_rows)) complete worker results, $(length(missing_results)) missing worker results, and two C_match masks.")
        return (; options, rows, long_rows, source_files, missing_results, c_match_masks)
    end
    mkpath(options.output)
    csv_path = write_csv(options.output, rows)
    markdown_path = write_markdown(options.output, rows, options.experiments)
    figure_paths = Dict{Symbol, Any}()
    for protocol in PAPER_PROTOCOLS
        paths = write_protocol_plot(options.output, rows, protocol)
        isnothing(paths) || (figure_paths[protocol] = paths)
    end
    c_match_mask_paths = write_c_match_mask_plot(options.output, c_match_masks)
    provenance_path = write_provenance(options.output, source_files, options.experiments)
    metrics_path = atomic_save(
        joinpath(options.output, "paper_metrics.jld2");
        schema_version = NOISE_SCHEMA_VERSION,
        experiment = :package10_sensor_noise_paper_table,
        experiment_ids = Dict(string(protocol) => identifier for (protocol, identifier) in options.experiments),
        noise_levels = collect(NOISE_LEVELS),
        rows,
        long_rows,
        missing_results,
        csv_path,
        markdown_path,
        figure_paths,
        c_match_mask_paths,
        provenance_path,
        created_at = string(Dates.now(Dates.UTC)),
    )
    println("Package-10 paper figures and tables written to $(options.output)")
    println("  CSV: $csv_path")
    println("  Markdown: $markdown_path")
    for protocol in PAPER_PROTOCOLS
        haskey(figure_paths, protocol) || continue
        println("  $(PAPER_PROTOCOL_LABELS[protocol]) SVG: $(figure_paths[protocol].svg_path)")
        println("  $(PAPER_PROTOCOL_LABELS[protocol]) PDF: $(figure_paths[protocol].pdf_path)")
    end
    println("  C_match masks SVG: $(c_match_mask_paths.svg_path)")
    println("  C_match masks PDF: $(c_match_mask_paths.pdf_path)")
    isempty(missing_results) || println("  Missing worker results represented as NA: $(length(missing_results))")
    return (; options, rows, long_rows, csv_path, markdown_path, figure_paths, c_match_mask_paths, provenance_path, metrics_path)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
