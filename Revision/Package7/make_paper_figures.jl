ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using JLD2
using PlotlyJS
using Printf
using SHA

include(joinpath(@__DIR__, "Package7Study.jl"))
using .Package7Study

const PAPER_METHODS = ("go", "gr", "group-lasso", "growl")
const PAPER_GROUPINGS = ("gc", "sc")
const PAPER_METHOD_NAMES = Dict(
    "go" => "GO", "gr" => "GR", "group-lasso" => "Group Lasso", "growl" => "GrOWL",
)
const PAPER_THRESHOLD_COLORS = ("#2166AC", "#92C5DE", "#D6604D", "#67001F")
const PAPER_CHANNEL_COLORS = ("#277DA1", "#F2A13A", "#B41A5C")
const PAPER_CHANNEL_NAMES = ("Buoyancy b", "Vertical velocity w", "Horizontal velocity u")
const PAPER_INACTIVE_COLOR = "#F2F2F2"
const PAPER_GRID_COLOR = "#E6E6E6"
const DEFAULT_P7_RESULTS = joinpath(@__DIR__, "results")
const STRIPE_CHANNEL_WIDTH = 4
const STRIPE_SENSOR_WIDTH = 3 * STRIPE_CHANNEL_WIDTH + 1
const STRIPE_COLUMN_COUNT = 48 * STRIPE_SENSOR_WIDTH - 1

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/Package7/make_paper_figures.jl \\
        [--experiment-id ID] [--results-dir PATH] [--output-dir PATH] [--check-only]

    The script reads completed Package-7 analyses and existing Fixed baseline
    artifacts. Without --experiment-id it uses the newest direct directory in
    the results root. It never selects a new candidate or executes a rollout.
    """)
end

function parse_arguments(arguments)
    values = Dict{String, Any}(
        "experiment_id" => nothing,
        "results_dir" => DEFAULT_P7_RESULTS,
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
    results_root = abspath(string(values["results_dir"]))
    experiment_id = isnothing(values["experiment_id"]) ?
        latest_experiment_id(results_root) : normalize_experiment_id(values["experiment_id"])
    output = isnothing(values["output_dir"]) ?
        joinpath(results_root, experiment_id, "paper") : abspath(string(values["output_dir"]))
    return (; experiment_id, results_root, output, check_only = Bool(values["check_only"]))
end

function latest_experiment_id(results_root)
    isdir(results_root) || error("Package-7 results directory is missing: $results_root")
    directories = [
        joinpath(results_root, name) for name in readdir(results_root)
        if isdir(joinpath(results_root, name)) &&
           !startswith(name, ".") &&
           any(configuration -> isdir(joinpath(results_root, name, configuration)), P7_CONFIGURATION_NAMES) &&
           try
               normalize_experiment_id(name)
               true
           catch
               false
           end
    ]
    isempty(directories) && error("No Package-7 experiment directories found below $results_root.")
    timestamped = filter(path -> occursin(r"^\d{6}_\d{6}$", basename(path)), directories)
    newest = isempty(timestamped) ?
        last(sort(directories; by = path -> (stat(path).mtime, basename(path)))) :
        last(sort(timestamped; by = basename))
    identifier = normalize_experiment_id(basename(newest))
    println("No --experiment-id supplied; using newest Package-7 results directory: $identifier")
    return identifier
end

function split_csv_line(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    quoted = false
    index = firstindex(line)
    while index <= lastindex(line)
        character = line[index]
        if character == '"'
            next_index = nextind(line, index)
            if quoted && next_index <= lastindex(line) && line[next_index] == '"'
                write(buffer, '"')
                index = nextind(line, next_index)
                continue
            end
            quoted = !quoted
        elseif character == ',' && !quoted
            push!(fields, String(take!(buffer)))
        else
            write(buffer, character)
        end
        index = nextind(line, index)
    end
    quoted && error("Unterminated quoted CSV field.")
    push!(fields, String(take!(buffer)))
    return fields
end

function read_csv(path)
    isfile(path) || error("Required Package-7 CSV is missing: $path")
    lines = readlines(path)
    isempty(lines) && error("CSV is empty: $path")
    headers = Symbol.(split_csv_line(lines[1]))
    rows = Dict{Symbol, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split_csv_line(line)
        length(values) == length(headers) || error("Malformed CSV row in $path")
        push!(rows, Dict(headers[index] => values[index] for index in eachindex(headers)))
    end
    return rows
end

string_value(row, key) = string(row[key])
int_value(row, key) = parse(Int, string(row[key]))
float_value(row, key) = parse(Float64, string(row[key]))

function normalize_record(record)
    return Dict{Symbol, Any}(Symbol(key) => value for (key, value) in pairs(record))
end

file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function analysis_paths(options, configuration)
    root = analysis_directory(options.results_root, options.experiment_id, configuration)
    return (;
        root,
        status = joinpath(root, "status.jld2"),
        evaluations = joinpath(root, "evaluations.csv"),
        front = joinpath(root, "pooled_pareto_front.csv"),
        selection = joinpath(root, "selected_test_candidate.jld2"),
        test = joinpath(root, "test", "test_results.jld2"),
    )
end

function load_configuration(options, configuration)
    paths = analysis_paths(options, configuration)
    all(isfile, (paths.status, paths.evaluations, paths.front)) || error(
        "$configuration analysis is incomplete below $(paths.root).",
    )
    status = JLD2.load(paths.status)
    Symbol(status["state"]) === :complete || error("$configuration analysis status is not complete.")
    string(status["experiment_id"]) == options.experiment_id || error("$configuration experiment mismatch.")
    string(status["configuration"]) == configuration || error("$configuration status mismatch.")
    evaluations = read_csv(paths.evaluations)
    front = read_csv(paths.front)
    qualified_front = filter(
        row -> float_value(row, :validation_matching) <= P7_QUALITY_THRESHOLD,
        front,
    )
    selected = nothing
    test = nothing
    if isfile(paths.selection) || isfile(paths.test)
        isfile(paths.selection) && isfile(paths.test) || error(
            "$configuration has only one of selection and test result.",
        )
        selection = JLD2.load(paths.selection)
        Bool(selection["frozen_before_test"]) || error("$configuration selection was not frozen before test.")
        selection["selection_uses_test_data"] == false || error("$configuration selection used test data.")
        Float64(selection["quality_threshold"]) == P7_QUALITY_THRESHOLD || error("$configuration quality threshold mismatch.")
        selected = normalize_record(selection["candidate"])
        test = JLD2.load(paths.test)
        isempty(qualified_front) && error("$configuration stores a test candidate although its pooled front has no qualified point.")
        expected = first(sort(qualified_front; by = row -> (
            int_value(row, :active_inputs),
            int_value(row, :active_groups),
            float_value(row, :validation_matching),
            int_value(row, :update),
            string_value(row, :run_id),
            string_value(row, :candidate_id),
        )))
        string(selected[:candidate_id]) == string_value(expected, :candidate_id) || error(
            "$configuration frozen test candidate does not match the sparsest qualified pooled-front point.",
        )
        string(test["candidate_id"]) == string(selected[:candidate_id]) || error("$configuration test candidate mismatch.")
        Int(test["active_inputs"]) == Int(selected[:active_inputs]) || error("$configuration active-input mismatch.")
        Float64(test["validation_matching"]) <= P7_QUALITY_THRESHOLD || error("$configuration selected candidate exceeds quality threshold.")
        length(test["state_nusselt"]) == 200 || error("$configuration test does not contain 200 state_Nu values.")
        mask = BitArray(selected[:global_mask])
        size(mask) == (3, 48, 8) || error("$configuration global mask has size $(size(mask)), expected (3, 48, 8).")
        count(mask) == Int(selected[:active_inputs]) || error("$configuration global mask/active-input count mismatch.")
        selected[:global_mask] = mask
    elseif !isempty(qualified_front)
        error("$configuration has qualified pooled-front points but no frozen test result; rerun its Package-7 analyzer.")
    end
    native = filter(row ->
        string_value(row, :threshold_id) == "native" &&
        float_value(row, :validation_matching) <= P7_QUALITY_THRESHOLD,
        evaluations,
    )
    minimum_native_groups = isempty(native) ? missing : minimum(int_value(row, :active_groups) for row in native)
    return (; configuration, paths, status, evaluations, front, selected, test, minimum_native_groups)
end

function baseline_root()
    return abspath(get(
        ENV,
        "REVISION_BASELINE_RESULTS_DIR",
        joinpath(@__DIR__, "..", "Baselines", "results"),
    ))
end

function load_fixed_baseline(controller)
    path = joinpath(baseline_root(), "fixed", "$controller.jld2")
    isfile(path) || error("Fixed $controller baseline is missing: $path")
    loaded = JLD2.load(path)
    string(loaded["status"]) == "complete" || error("Fixed $controller baseline is incomplete.")
    Symbol(loaded["protocol"]) === :fixed || error("Fixed $controller baseline protocol mismatch.")
    Symbol(loaded["controller"]) === controller || error("Fixed $controller baseline controller mismatch.")
    Int(loaded["case_count"]) == 1 || error("Fixed $controller baseline must contain one episode.")
    episode = only(loaded["episodes"])
    length(episode.state_nusselt) == 200 || error("Fixed $controller baseline must contain 200 state_Nu values.")
    return (; controller, path, loaded, episode)
end

function validate_expert_identity(configurations, expert_baseline)
    identifiers = unique(
        string(data.test["expert_identifier"]) for data in values(configurations)
        if !isnothing(data.test)
    )
    length(identifiers) <= 1 || error("Package-7 analyses use different experts.")
    if !isempty(identifiers)
        expected = replace(only(identifiers), r"^sha256:" => "")
        string(expert_baseline.loaded["expert_sha256"]) == expected || error(
            "Fixed expert baseline does not match the Package-7 expert.",
        )
    end
    return nothing
end

function selected_sparsities(selected)
    mask = selected[:global_mask]
    active_channel_inputs = count(mask)
    occupied_locations = count(dropdims(any(mask; dims = 1); dims = 1))
    return (;
        sc = 100 * (1 - active_channel_inputs / (8 * 48 * 3)),
        gc = 100 * (1 - occupied_locations / (8 * 48)),
    )
end

function table_rows(configurations, expert, unactuated)
    rows = NamedTuple[(;
        configuration = "Full sensor set expert",
        active_groups = missing,
        global_sc_sparsity_percent = missing,
        global_gc_sparsity_percent = missing,
        validation_mse = missing,
        strength = missing,
        mask_threshold = missing,
        sum_state_nusselt = Float64(expert.episode.sum_state_nusselt),
        minimum_native_active_groups_under_quality_threshold = missing,
    )]
    for configuration in P7_CONFIGURATION_NAMES
        data = configurations[configuration]
        if isnothing(data.selected)
            push!(rows, (;
                configuration,
                active_groups = missing,
                global_sc_sparsity_percent = missing,
                global_gc_sparsity_percent = missing,
                validation_mse = missing,
                strength = missing,
                mask_threshold = missing,
                sum_state_nusselt = missing,
                minimum_native_active_groups_under_quality_threshold = data.minimum_native_groups,
            ))
            continue
        end
        sparsity = selected_sparsities(data.selected)
        push!(rows, (;
            configuration,
            active_groups = Int(data.selected[:active_groups]),
            global_sc_sparsity_percent = sparsity.sc,
            global_gc_sparsity_percent = sparsity.gc,
            validation_mse = Float64(data.selected[:validation_matching]),
            strength = Float64(data.selected[:regularization_strength]),
            mask_threshold = Float64(data.selected[:threshold_value]),
            sum_state_nusselt = Float64(data.test["sum_state_nusselt"]),
            minimum_native_active_groups_under_quality_threshold = data.minimum_native_groups,
        ))
    end
    push!(rows, (;
        configuration = "Unactuated",
        active_groups = missing,
        global_sc_sparsity_percent = missing,
        global_gc_sparsity_percent = missing,
        validation_mse = missing,
        strength = missing,
        mask_threshold = missing,
        sum_state_nusselt = Float64(unactuated.episode.sum_state_nusselt),
        minimum_native_active_groups_under_quality_threshold = missing,
    ))
    return rows
end

csv_value(value) = value === missing ? "" : string(value)

function write_table(output, rows)
    csv_path = joinpath(output, "table_1_selected_candidates.csv")
    markdown_path = joinpath(output, "table_1_selected_candidates.md")
    headers = (
        :configuration, :active_groups, :global_sc_sparsity_percent,
        :global_gc_sparsity_percent, :validation_mse, :strength,
        :mask_threshold, :sum_state_nusselt,
        :minimum_native_active_groups_under_quality_threshold,
    )
    open(csv_path, "w") do io
        println(io, join(string.(headers), ','))
        for row in rows
            println(io, join((csv_value(getproperty(row, key)) for key in headers), ','))
        end
    end
    fmt(value, format) = value === missing ? "" : Printf.format(Printf.Format(format), value)
    open(markdown_path, "w") do io
        println(io, "# Package 7 selected candidates\n")
        println(io, "| Configuration | Active groups | Global SC sparsity | Global GC sparsity | Validation MSE | Strength | Mask threshold | Test sum(state_Nu) | Minimum native groups under quality threshold |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows
            println(io, "| $(row.configuration) | $(fmt(row.active_groups, "%d")) | $(fmt(row.global_sc_sparsity_percent, "%.2f%%")) | $(fmt(row.global_gc_sparsity_percent, "%.2f%%")) | $(fmt(row.validation_mse, "%.4e")) | $(fmt(row.strength, "%.6g")) | $(fmt(row.mask_threshold, "%.6g")) | $(fmt(row.sum_state_nusselt, "%.6f")) | $(fmt(row.minimum_native_active_groups_under_quality_threshold, "%d")) |")
        end
        println(io, "\nQuality means validation MSE <= $(P7_QUALITY_THRESHOLD). SC sparsity uses 8×48×3 channel inputs; GC sparsity treats a location as occupied when any channel is active. The final column is the only candidate-independent measurement.")
    end
    return (; csv_path, markdown_path)
end

function panel_titles(methods = PAPER_METHODS)
    return reshape([
        "$(PAPER_METHOD_NAMES[method]) - $(uppercase(grouping))"
        for method in methods for grouping in PAPER_GROUPINGS
    ], :, 1)
end

function stripe_matrix(mask)
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
    return attr(; fields...)
end

axis_key(axis, index) = Symbol(index == 1 ? axis : "$(axis)$(index)")

function make_mask_figure(
    configurations,
    output;
    methods = PAPER_METHODS,
    stem = "figure_1_selected_sensor_masks",
    title = "Package 7: selected global input masks",
)
    row_count = length(methods)
    height = row_count == 4 ? 1550 : 850
    plot = make_subplots(rows = row_count, cols = 2, vertical_spacing = row_count == 4 ? 0.055 : 0.10, horizontal_spacing = 0.07, subplot_titles = panel_titles(methods))
    for (row, method) in enumerate(methods), (col, grouping) in enumerate(PAPER_GROUPINGS)
        data = configurations["$method-$grouping"]
        if isnothing(data.selected)
            add_trace!(plot, scatter(x = [72], y = [4.5], mode = "text", text = ["NR"], textfont = attr(size = 22, color = "#777777"), showlegend = false); row, col)
        else
            values, text = stripe_matrix(data.selected[:global_mask])
            add_trace!(plot, heatmap(
                x = collect(1:STRIPE_COLUMN_COUNT), y = collect(1:8),
                z = values, text = text, zmin = 0, zmax = 3,
                colorscale = STRIPE_COLORSCALE, showscale = false,
                hovertemplate = "%{text}<extra></extra>",
                xgap = 0, ygap = 1,
            ); row, col)
        end
    end
    for (index, (name, color)) in enumerate(zip(("Inactive", PAPER_CHANNEL_NAMES...), (PAPER_INACTIVE_COLOR, PAPER_CHANNEL_COLORS...)))
        add_trace!(plot, scatter(
            x = [NaN], y = [NaN], mode = "markers", name = name,
            marker = attr(color = color, size = 11, symbol = "square", line = attr(color = "#444444", width = index == 1 ? 1 : 0)),
            legendgroup = "mask_legend", showlegend = true,
        ); row = 1, col = 1)
    end
    layout = Dict{Symbol, Any}(
        :template => "plotly_white", :width => 1450, :height => height,
        :title => attr(text = title, x = 0.5, xanchor = "center"),
        :paper_bgcolor => "white", :plot_bgcolor => "white",
        :font => attr(family = "Arial, sans-serif", size = 13, color = "#303030"),
        :margin => attr(l = 80, r = 30, t = 100, b = 100),
        :legend => attr(orientation = "h", x = 0.5, xanchor = "center", y = row_count == 4 ? -0.055 : -0.105, yanchor = "top"),
    )
    for index in 1:(2 * row_count)
        layout[axis_key("xaxis", index)] = preserved_axis(plot, axis_key("xaxis", index), attr(
            title = index > 2 * (row_count - 1) ? "Horizontal sensor index" : "",
            range = [0.5, STRIPE_COLUMN_COUNT + 0.5], tickmode = "array",
            tickvals = [stripe_sensor_center(value) for value in 1:4:48],
            ticktext = string.(1:4:48), showgrid = false, zeroline = false,
            showline = true, mirror = true, linecolor = "#3A3A3A",
        ))
        layout[axis_key("yaxis", index)] = preserved_axis(plot, axis_key("yaxis", index), attr(
            title = isodd(index) ? "Vertical sensor index" : "",
            range = [0.5, 8.5], tickmode = "array", tickvals = collect(1:8),
            showgrid = false, zeroline = false, showline = true, mirror = true, linecolor = "#3A3A3A",
        ))
    end
    relayout!(plot, layout)
    paths = String[]
    for extension in ("svg", "pdf")
        path = joinpath(output, "$stem.$extension")
        PlotlyJS.savefig(plot, path; width = 1450, height = height)
        push!(paths, path)
    end
    return paths
end

function threshold_colors(configurations)
    thresholds = sort!(unique(reduce(vcat, [
        [float_value(row, :threshold_value) for row in data.evaluations]
        for data in values(configurations)
    ])))
    return thresholds, Dict(value => PAPER_THRESHOLD_COLORS[mod1(index, length(PAPER_THRESHOLD_COLORS))] for (index, value) in enumerate(thresholds))
end

function move_glimages_behind_cartesian!(path)
    svg = read(path, String)
    matches = collect(eachmatch(r"<g class=\"glimages\">.*?</g>"s, svg))
    length(matches) == 1 || error("Expected one glimages layer in $path, found $(length(matches)).")
    layer = only(matches).match
    without = replace(svg, layer => ""; count = 1)
    marker = "<g class=\"cartesianlayer\">"
    occursin(marker, without) || error("Cartesian SVG layer is missing in $path.")
    temporary = path * ".tmp"
    write(temporary, replace(without, marker => layer * marker; count = 1))
    mv(temporary, path; force = true)
    return path
end

function make_pareto_figure(configurations, output)
    thresholds, colors = threshold_colors(configurations)
    plot = make_subplots(rows = 4, cols = 2, vertical_spacing = 0.055, horizontal_spacing = 0.07, subplot_titles = panel_titles())
    shapes = Any[]
    all_losses = Float64[]
    maxima = Dict(grouping => 1 for grouping in PAPER_GROUPINGS)
    shown_thresholds = Set{Float64}()
    for (row, method) in enumerate(PAPER_METHODS), (col, grouping) in enumerate(PAPER_GROUPINGS)
        data = configurations["$method-$grouping"]
        index = 2 * (row - 1) + col
        for threshold in thresholds
            selected = filter(item ->
                float_value(item, :threshold_value) == threshold &&
                isfinite(float_value(item, :validation_matching)) &&
                float_value(item, :validation_matching) > 0,
                data.evaluations,
            )
            isempty(selected) && continue
            append!(all_losses, float_value.(selected, Ref(:validation_matching)))
            maxima[grouping] = max(maxima[grouping], maximum(int_value(item, :active_groups) for item in selected))
            showlegend = !(threshold in shown_thresholds)
            showlegend && push!(shown_thresholds, threshold)
            add_trace!(plot, scattergl(
                x = int_value.(selected, Ref(:active_groups)),
                y = float_value.(selected, Ref(:validation_matching)),
                mode = "markers", name = "τ=$(threshold)", showlegend = showlegend,
                legendgroup = "threshold_$threshold",
                marker = attr(
                    color = colors[threshold], size = 4, opacity = 0.32,
                    symbol = [("circle", "diamond", "square")[int_value(item, :replicate)] for item in selected],
                ),
                customdata = hcat(
                    int_value.(selected, Ref(:active_inputs)),
                    int_value.(selected, Ref(:replicate)),
                    float_value.(selected, Ref(:strength)),
                    int_value.(selected, Ref(:update)),
                ),
                hovertemplate = "groups=%{x}<br>inputs=%{customdata[0]}<br>MSE=%{y:.4e}<br>replicate=%{customdata[1]}<br>strength=%{customdata[2]:.4g}<br>update=%{customdata[3]}<extra></extra>",
            ); row, col)
        end
        front = sort(filter(item -> float_value(item, :validation_matching) > 0, data.front); by = item -> int_value(item, :active_inputs))
        add_trace!(plot, scatter(
            x = int_value.(front, Ref(:active_groups)),
            y = float_value.(front, Ref(:validation_matching)),
            mode = "lines+markers", name = "Pooled Pareto front",
            legendgroup = "pooled_front", showlegend = index == 1,
            line = attr(color = "#111111", width = 2.2), marker = attr(color = "#111111", size = 6, symbol = "circle-open"),
        ); row, col)
        if !isnothing(data.selected)
            add_trace!(plot, scatter(
                x = [Int(data.selected[:active_groups])],
                y = [Float64(data.selected[:validation_matching])],
                mode = "markers", name = "Selected test candidate",
                legendgroup = "selected", showlegend = index == 1,
                marker = attr(color = "#F2C14E", size = 14, symbol = "star", line = attr(color = "#111111", width = 1.2)),
            ); row, col)
        end
        axis_suffix = index == 1 ? "" : string(index)
        push!(shapes, attr(
            type = "line", xref = "x$axis_suffix domain", x0 = 0, x1 = 1,
            yref = "y$axis_suffix", y0 = P7_QUALITY_THRESHOLD, y1 = P7_QUALITY_THRESHOLD,
            line = attr(color = "#555555", width = 1.3, dash = "dash"),
        ))
    end
    isempty(all_losses) && error("No finite Package-7 evaluation losses were found.")
    y_range = [log10(minimum(all_losses)) - 0.15, log10(10.0)]
    layout = Dict{Symbol, Any}(
        :template => "plotly_white", :width => 1450, :height => 1650,
        :title => attr(text = "Package 7: evaluation landscapes and pooled Pareto fronts", x = 0.5, xanchor = "center"),
        :paper_bgcolor => "white", :plot_bgcolor => "white", :shapes => shapes,
        :font => attr(family = "Arial, sans-serif", size = 13, color = "#303030"),
        :margin => attr(l = 85, r = 30, t = 100, b = 110),
        :legend => attr(orientation = "h", x = 0.5, xanchor = "center", y = -0.06, yanchor = "top"),
    )
    for index in 1:8
        grouping = isodd(index) ? "gc" : "sc"
        layout[axis_key("xaxis", index)] = preserved_axis(plot, axis_key("xaxis", index), attr(
            title = index > 6 ? "Active groups" : "", range = [-0.5, maxima[grouping] + 1],
            showline = true, mirror = true, linecolor = "#3A3A3A", ticks = "outside",
            gridcolor = PAPER_GRID_COLOR, zeroline = false,
        ))
        layout[axis_key("yaxis", index)] = preserved_axis(plot, axis_key("yaxis", index), attr(
            title = isodd(index) ? "Validation MSE" : "", type = "log", range = y_range,
            showline = true, mirror = true, linecolor = "#3A3A3A", ticks = "outside",
            gridcolor = PAPER_GRID_COLOR, zeroline = false,
        ))
    end
    relayout!(plot, layout)
    svg_path = joinpath(output, "figure_s1_pareto_comparison.svg")
    pdf_path = joinpath(output, "figure_s1_pareto_comparison.pdf")
    PlotlyJS.savefig(plot, svg_path; width = 1450, height = 1650)
    move_glimages_behind_cartesian!(svg_path)
    PlotlyJS.savefig(plot, pdf_path; width = 1450, height = 1650)
    return [svg_path, pdf_path]
end

function write_provenance(output, configurations, expert, unactuated)
    files = String[expert.path, unactuated.path]
    for data in values(configurations)
        append!(files, [data.paths.status, data.paths.evaluations, data.paths.front])
        isfile(data.paths.selection) && push!(files, data.paths.selection)
        isfile(data.paths.test) && push!(files, data.paths.test)
    end
    sort!(unique!(files))
    path = joinpath(output, "provenance.sha256")
    open(path, "w") do io
        println(io, "# Package-7 paper input manifest")
        println(io, "# Script SHA-256: $(file_sha256(abspath(@__FILE__)))")
        for file in files
            println(io, "$(file_sha256(file))  $(replace(relpath(file, output), '\\' => '/'))")
        end
    end
    return path
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    configurations = Dict(
        configuration => load_configuration(options, configuration)
        for configuration in P7_CONFIGURATION_NAMES
    )
    expert = load_fixed_baseline(:expert)
    unactuated = load_fixed_baseline(:unactuated)
    validate_expert_identity(configurations, expert)
    rows = table_rows(configurations, expert, unactuated)
    if options.check_only
        qualified = count(data -> !isnothing(data.selected), values(configurations))
        println("Package-7 paper inputs valid: $qualified/8 qualified selected candidates.")
        return (; options, configurations, expert, unactuated, rows)
    end
    mkpath(options.output)
    table = write_table(options.output, rows)
    mask_paths = vcat(
        make_mask_figure(configurations, options.output),
        make_mask_figure(
            configurations,
            options.output;
            methods = ("go", "gr"),
            stem = "figure_1a_selected_sensor_masks_go_gr",
            title = "Package 7: GO and GR selected global input masks",
        ),
        make_mask_figure(
            configurations,
            options.output;
            methods = ("group-lasso", "growl"),
            stem = "figure_1b_selected_sensor_masks_group_lasso_growl",
            title = "Package 7: Group Lasso and GrOWL selected global input masks",
        ),
    )
    pareto_paths = make_pareto_figure(configurations, options.output)
    provenance = write_provenance(options.output, configurations, expert, unactuated)
    atomic_save(
        joinpath(options.output, "paper_metrics.jld2");
        schema_version = P7_SCHEMA_VERSION,
        experiment_id = options.experiment_id,
        quality_threshold = P7_QUALITY_THRESHOLD,
        table_rows = rows,
        table,
        mask_paths,
        pareto_paths,
        provenance,
    )
    println("Package-7 paper artifacts written to $(options.output)")
    return (; options, configurations, rows, table, mask_paths, pareto_paths, provenance)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
