ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using JLD2
using PlotlyJS
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "HigherRaGOStudy.jl"))
using .HigherRaGOStudy

const PAPER_STUDIES = (:ra5e4, :ra1e5)
const PAPER_METHODS = ("go", "gr")
const PAPER_GROUPINGS = ("gc", "sc")
const PAPER_METHOD_NAMES = Dict("go" => "GO", "gr" => "GR")
const PAPER_ZERO_THRESHOLD_COLOR = "#277DA1"
const PAPER_THRESHOLD_COLORS = ("#F2A13A", "#E6782B", "#B2182B")
const PAPER_QUALITY_COLORS = ("#6A4C93", "#2A9D8F", "#D1495B")
const PAPER_QUALITY_SYMBOLS = ("star", "diamond", "x")
const PAPER_CHANNEL_COLORS = ("#277DA1", "#F2A13A", "#B41A5C")
const PAPER_CHANNEL_NAMES = ("Buoyancy b", "Vertical velocity w", "Horizontal velocity u")
const PAPER_INACTIVE_COLOR = "#F2F2F2"
const PAPER_GRID_COLOR = "#E6E6E6"
const STRIPE_CHANNEL_WIDTH = 4
const STRIPE_SENSOR_WIDTH = 3 * STRIPE_CHANNEL_WIDTH + 1
const STRIPE_COLUMN_COUNT = 48 * STRIPE_SENSOR_WIDTH - 1
const DEFAULT_NEAR_EXPERT_RELATIVE_TOLERANCE = 0.05

struct PaperPoint
    run_id::String
    candidate_id::String
    replicate::Int16
    strength::Float64
    update::Int32
    threshold_id::Symbol
    threshold_value::Float64
    active_groups::Int16
    active_inputs::Int16
    validation_mse::Float64
end

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. \\
        Revision/Higher_Ra_Study/GO_GR_Study/make_paper_figures.jl \\
        [--study all|ra5e4|ra1e5] [--experiment-id ID] \\
        [--results-dir PATH] [--output-dir PATH] \\
        [--near-expert-relative-tolerance 0.05] [--check-only]

    The script reads completed Higher-Ra GO/GR analyses and the matching local
    expert/unactuated baselines. Without --experiment-id it independently uses
    the newest direct experiment directory for each selected Rayleigh number.
    It never trains a model, selects a validation candidate, or runs a rollout.

    For the sensor-mask figure, "near expert" means a test mean(state_Nu) no
    more than the configured relative tolerance above the expert mean. Lower
    state_Nu is better. The default tolerance is 5%.
    """)
end

function parse_arguments(arguments)
    parsed = Dict{String, Any}(
        "study" => "all",
        "experiment_id" => nothing,
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "output_dir" => nothing,
        "near_expert_relative_tolerance" => DEFAULT_NEAR_EXPERT_RELATIVE_TOLERANCE,
        "check_only" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--check-only"
            parsed["check_only"] = true
            index += 1
        elseif startswith(argument, "--")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(parsed, key) || error("Unknown option '$argument'.")
            parsed[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument '$argument'.")
        end
    end
    selected_studies = lowercase(strip(string(parsed["study"]))) == "all" ?
        collect(PAPER_STUDIES) : [normalize_study(parsed["study"])]
    tolerance = parse(Float64, string(parsed["near_expert_relative_tolerance"]))
    isfinite(tolerance) && tolerance >= 0 || error(
        "--near-expert-relative-tolerance must be finite and nonnegative.",
    )
    results_root = abspath(string(parsed["results_dir"]))
    experiment_ids = Dict{Symbol, String}()
    if isnothing(parsed["experiment_id"])
        for study_tag in selected_studies
            experiment_ids[study_tag] = latest_experiment_id(results_root, study_tag)
        end
        if length(unique(collect(Base.values(experiment_ids)))) > 1
            @warn "Newest Higher-Ra experiment IDs differ; continuing independently." experiment_ids
        end
    else
        experiment_id = normalize_experiment_id(parsed["experiment_id"])
        for study_tag in selected_studies
            experiment_ids[study_tag] = experiment_id
        end
    end
    output_root = isnothing(parsed["output_dir"]) ? nothing : abspath(string(parsed["output_dir"]))
    return (;
        studies = selected_studies,
        experiment_ids,
        results_root,
        output_root,
        near_expert_relative_tolerance = tolerance,
        check_only = Bool(parsed["check_only"]),
    )
end

function latest_experiment_id(results_root, study_tag)
    root = joinpath(results_root, string(study_tag))
    isdir(root) || error("Higher-Ra results directory is missing: $root")
    directories = [
        joinpath(root, name) for name in readdir(root)
        if isdir(joinpath(root, name)) &&
           !startswith(name, ".") &&
           any(configuration_name ->
               isdir(joinpath(root, name, configuration_name)), HR_CONFIGURATION_NAMES) &&
           try
               normalize_experiment_id(name)
               true
           catch
               false
           end
    ]
    isempty(directories) && error("No $(study(study_tag).label) experiment directories found below $root.")
    timestamped = filter(path -> occursin(r"^\d{6}_\d{6}$", basename(path)), directories)
    newest = isempty(timestamped) ?
        last(sort(directories; by = path -> (stat(path).mtime, basename(path)))) :
        last(sort(timestamped; by = basename))
    identifier = normalize_experiment_id(basename(newest))
    println("No --experiment-id supplied; using newest $(study(study_tag).label) results directory: $identifier")
    return identifier
end

function output_directory(options, study_tag)
    if isnothing(options.output_root)
        return joinpath(
            options.results_root,
            string(study_tag),
            options.experiment_ids[study_tag],
            "paper",
        )
    end
    return length(options.studies) == 1 ? options.output_root :
        joinpath(options.output_root, string(study_tag))
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

function read_points(path; keep_identity = false)
    isfile(path) || error("Required Higher-Ra CSV is missing: $path")
    stream = eachline(path)
    first_line = iterate(stream)
    isnothing(first_line) && error("CSV is empty: $path")
    header_line, state = first_line
    headers = Symbol.(split_csv_line(header_line))
    positions = Dict(header => index for (index, header) in enumerate(headers))
    required = (
        :run_id, :candidate_id, :replicate, :strength, :update, :threshold_id,
        :threshold_value, :active_groups, :active_inputs, :validation_matching,
    )
    all(haskey(positions, key) for key in required) || error("Unexpected CSV schema in $path")
    points = PaperPoint[]
    current = iterate(stream, state)
    while !isnothing(current)
        line, state = current
        if !isempty(strip(line))
            fields = split_csv_line(line)
            length(fields) == length(headers) || error("Malformed CSV row in $path")
            value(key) = fields[positions[key]]
            push!(points, PaperPoint(
                keep_identity ? value(:run_id) : "",
                keep_identity ? value(:candidate_id) : "",
                Int16(parse(Int, value(:replicate))),
                parse(Float64, value(:strength)),
                Int32(parse(Int, value(:update))),
                Symbol(value(:threshold_id)),
                parse(Float64, value(:threshold_value)),
                Int16(parse(Int, value(:active_groups))),
                Int16(parse(Int, value(:active_inputs))),
                parse(Float64, value(:validation_matching)),
            ))
        end
        current = iterate(stream, state)
    end
    return points
end

normalize_record(record) = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in pairs(record))
file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function analysis_paths(options, study_tag, configuration_name)
    root = analysis_directory(
        options.results_root,
        study_tag,
        options.experiment_ids[study_tag],
        configuration_name,
    )
    return (;
        root,
        status = joinpath(root, "status.jld2"),
        evaluations = joinpath(root, "evaluations.csv"),
        front = joinpath(root, "pooled_pareto_front.csv"),
        selection = joinpath(root, "selected_test_candidates.jld2"),
    )
end

candidate_sort_key(point) = (
    Int(point.active_inputs), Int(point.active_groups), point.validation_mse,
    Int(point.update), point.run_id, point.candidate_id,
)

function same_quality_thresholds(values)
    observed = Float64.(values)
    length(observed) == length(HR_QUALITY_THRESHOLDS) || return false
    return all(isapprox(a, b; atol = 1e-12, rtol = 1e-10)
               for (a, b) in zip(observed, HR_QUALITY_THRESHOLDS))
end

function test_mean_state_nusselt(test_result)
    episodes = test_result["episodes"]
    return mean(Float64(value) for episode in episodes for value in episode.state_nusselt)
end

function load_configuration(options, study_tag, configuration_name)
    paths = analysis_paths(options, study_tag, configuration_name)
    all(isfile, (paths.status, paths.evaluations, paths.front, paths.selection)) || error(
        "$configuration_name analysis is incomplete below $(paths.root). " *
        "Rerun its Higher-Ra analysis worker.",
    )
    status = JLD2.load(paths.status)
    Symbol(status["state"]) === :complete || error(
        "$configuration_name analysis status is $(status["state"]), not complete. " *
        "Rerun its Higher-Ra analysis worker.",
    )
    Symbol(status["study"]) === study_tag || error("$configuration_name analysis study mismatch.")
    string(status["configuration"]) == configuration_name || error("$configuration_name status mismatch.")
    same_quality_thresholds(status["quality_thresholds"]) || error(
        "$configuration_name analysis uses unexpected quality thresholds.",
    )
    stored_experiment_id = string(status["experiment_id"])
    selected_experiment_id = options.experiment_ids[study_tag]
    if stored_experiment_id != selected_experiment_id
        @warn "$configuration_name experiment mismatch; continuing with relocated artifacts." selected_experiment_id stored_experiment_id analysis_directory=paths.root
    end

    evaluations = read_points(paths.evaluations)
    front = read_points(paths.front; keep_identity = true)
    selection = JLD2.load(paths.selection)
    Bool(selection["frozen_before_test"]) || error("$configuration_name selection was not frozen before test.")
    selection["selection_uses_test_data"] == false || error("$configuration_name selection used test data.")
    same_quality_thresholds(selection["quality_thresholds"]) || error(
        "$configuration_name selection uses unexpected quality thresholds.",
    )

    mappings = [normalize_record(entry) for entry in selection["threshold_selections"]]
    length(mappings) == length(HR_QUALITY_THRESHOLDS) || error(
        "$configuration_name has an unexpected threshold-selection count.",
    )
    frozen_candidates = Dict{Symbol, Any}[]
    for raw in selection["candidates"]
        frozen = normalize_record(raw)
        candidate = normalize_record(frozen[:candidate])
        mask = BitArray(candidate[:global_mask])
        size(mask) == (3, 48, 8) || error(
            "$configuration_name candidate has mask size $(size(mask)), expected (3, 48, 8).",
        )
        count(mask) == Int(candidate[:active_inputs]) || error(
            "$configuration_name candidate mask/active-input mismatch.",
        )
        candidate[:global_mask] = mask
        frozen[:candidate] = candidate
        frozen[:quality_thresholds] = Float64.(frozen[:quality_thresholds])
        index = Int(frozen[:candidate_index])
        test_path = joinpath(paths.root, "test", @sprintf("candidate_%02d", index), "test_results.jld2")
        isfile(test_path) || error("$configuration_name candidate $index test result is missing: $test_path")
        test_result = JLD2.load(test_path)
        string(test_result["candidate_id"]) == string(candidate[:candidate_id]) || error(
            "$configuration_name candidate $index test identity mismatch.",
        )
        string(test_result["configuration"]) == configuration_name || error(
            "$configuration_name candidate $index test configuration mismatch.",
        )
        Symbol(test_result["study"]) === study_tag || error(
            "$configuration_name candidate $index test study mismatch.",
        )
        test_result["selection_uses_test_data"] == false || error(
            "$configuration_name candidate $index test selection used test data.",
        )
        Int(test_result["case_count"]) == 8 || error(
            "$configuration_name candidate $index must contain eight test cases.",
        )
        episodes = test_result["episodes"]
        length(episodes) == 8 || error("$configuration_name candidate $index has incomplete episodes.")
        all(episode -> length(episode.state_nusselt) == 200, episodes) || error(
            "$configuration_name candidate $index episodes must contain 200 state_Nu values.",
        )
        all(episode -> Symbol(episode.split) === :test, episodes) || error(
            "$configuration_name candidate $index did not exclusively use the test split.",
        )
        frozen[:test_path] = test_path
        frozen[:test] = test_result
        frozen[:mean_state_nusselt] = test_mean_state_nusselt(test_result)
        push!(frozen_candidates, frozen)
    end

    candidate_by_index = Dict(Int(item[:candidate_index]) => item for item in frozen_candidates)
    referenced_indices = Set{Int}()
    for (mapping, quality_threshold) in zip(mappings, HR_QUALITY_THRESHOLDS)
        isapprox(Float64(mapping[:quality_threshold]), quality_threshold; atol = 1e-12, rtol = 1e-10) ||
            error("$configuration_name quality-threshold mapping order mismatch.")
        qualified = filter(point -> point.validation_mse <= quality_threshold, front)
        expected = isempty(qualified) ? nothing : first(sort(qualified; by = candidate_sort_key))
        if isnothing(expected)
            isnothing(mapping[:candidate_id]) && isnothing(mapping[:candidate_index]) || error(
                "$configuration_name stores a candidate for empty quality threshold $quality_threshold.",
            )
        else
            string(mapping[:candidate_id]) == expected.candidate_id || error(
                "$configuration_name candidate for q=$quality_threshold does not match the pooled-front rule.",
            )
            candidate_index = Int(mapping[:candidate_index])
            haskey(candidate_by_index, candidate_index) || error(
                "$configuration_name mapping references missing candidate $candidate_index.",
            )
            string(candidate_by_index[candidate_index][:candidate][:candidate_id]) == expected.candidate_id ||
                error("$configuration_name frozen candidate mismatch for q=$quality_threshold.")
            push!(referenced_indices, candidate_index)
        end
    end
    Set(keys(candidate_by_index)) == referenced_indices || error(
        "$configuration_name contains a frozen candidate not selected by a quality threshold.",
    )
    return (; configuration = configuration_name, paths, status, evaluations, front,
            selection, mappings, candidates = frozen_candidates)
end

function load_baseline(study_tag, controller)
    study_config = study(study_tag)
    path = controller === :expert ? study_config.expert_baseline : study_config.unactuated_baseline
    isfile(path) || error("$(study_config.label) $controller baseline is missing: $path")
    loaded = JLD2.load(path)
    string(loaded["status"]) == "complete" || error("$(study_config.label) $controller baseline is incomplete.")
    Symbol(loaded["protocol"]) === study_config.protocol || error(
        "$(study_config.label) $controller baseline protocol mismatch.",
    )
    isapprox(Float64(loaded["rayleigh"]), study_config.rayleigh) || error(
        "$(study_config.label) $controller baseline Rayleigh mismatch.",
    )
    Int(loaded["case_count"]) == 8 || error("$(study_config.label) $controller baseline must contain eight episodes.")
    episodes = loaded["episodes"]
    all(episode -> length(episode.state_nusselt) == 200, episodes) || error(
        "$(study_config.label) $controller baseline episodes must contain 200 state_Nu values.",
    )
    mean_state_nusselt = mean(Float64(value) for episode in episodes for value in episode.state_nusselt)
    return (; controller, path, loaded, episodes, mean_state_nusselt)
end

function validate_expert_identity(configurations, expert)
    hashes = unique(
        string(candidate[:test]["expert_sha256"])
        for data in Base.values(configurations) for candidate in data.candidates
    )
    length(hashes) <= 1 || error("Higher-Ra analyses use different experts.")
    if !isempty(hashes)
        string(expert.loaded["expert_sha256"]) == only(hashes) || error(
            "Expert baseline does not match the Higher-Ra candidate tests.",
        )
    end
    return nothing
end

function selected_sparsities(candidate)
    mask = candidate[:global_mask]
    active_channel_inputs = count(mask)
    occupied_locations = count(dropdims(any(mask; dims = 1); dims = 1))
    return (;
        sc = 100 * (1 - active_channel_inputs / (8 * 48 * 3)),
        gc = 100 * (1 - occupied_locations / (8 * 48)),
    )
end

quality_label(values) = join((@sprintf("%.4g", value) for value in Float64.(values)), ", ")
configuration_label(configuration_name, thresholds) =
    "$configuration_name (q <= $(quality_label(thresholds)))"

function table_rows(configurations, expert, unactuated)
    rows = NamedTuple[(;
        configuration = "Full sensor set expert",
        base_configuration = "",
        quality_thresholds = Float64[],
        active_groups = missing,
        global_sc_sparsity_percent = missing,
        global_gc_sparsity_percent = missing,
        validation_mse = missing,
        strength = missing,
        mask_threshold = missing,
        mean_state_nusselt = expert.mean_state_nusselt,
    )]
    for configuration_name in HR_CONFIGURATION_NAMES
        data = configurations[configuration_name]
        for frozen in sort(data.candidates; by = item -> Int(item[:candidate_index]))
            candidate = frozen[:candidate]
            thresholds = sort(Float64.(frozen[:quality_thresholds]); rev = true)
            sparsity = selected_sparsities(candidate)
            push!(rows, (;
                configuration = configuration_label(configuration_name, thresholds),
                base_configuration = configuration_name,
                quality_thresholds = thresholds,
                active_groups = Int(candidate[:active_groups]),
                global_sc_sparsity_percent = sparsity.sc,
                global_gc_sparsity_percent = sparsity.gc,
                validation_mse = Float64(candidate[:validation_matching]),
                strength = Float64(candidate[:regularization_strength]),
                mask_threshold = Float64(candidate[:threshold_value]),
                mean_state_nusselt = Float64(frozen[:mean_state_nusselt]),
            ))
        end
    end
    push!(rows, (;
        configuration = "Unactuated",
        base_configuration = "",
        quality_thresholds = Float64[],
        active_groups = missing,
        global_sc_sparsity_percent = missing,
        global_gc_sparsity_percent = missing,
        validation_mse = missing,
        strength = missing,
        mask_threshold = missing,
        mean_state_nusselt = unactuated.mean_state_nusselt,
    ))
    return rows
end

csv_value(value) = value === missing ? "" : string(value)

function active_groups_value(row)
    row.active_groups === missing && return ""
    total = endswith(row.base_configuration, "-gc") ? 32 :
            endswith(row.base_configuration, "-sc") ? 96 :
            error("Cannot determine total group count for $(row.configuration).")
    return "$(row.active_groups)/$total"
end

table_value(row, key) = key === :active_groups ? active_groups_value(row) :
    csv_value(getproperty(row, key))

function write_table(output, rows, study_tag)
    stem = "table_1_selected_candidates_$(study_tag)"
    csv_path = joinpath(output, "$stem.csv")
    markdown_path = joinpath(output, "$stem.md")
    headers = (
        :configuration, :active_groups, :global_sc_sparsity_percent,
        :global_gc_sparsity_percent, :validation_mse, :strength,
        :mask_threshold, :mean_state_nusselt,
    )
    open(csv_path, "w") do io
        println(io, join(string.(headers), ','))
        for row in rows
            println(io, join((table_value(row, key) for key in headers), ','))
        end
    end
    fmt(value, format) = value === missing ? "" : Printf.format(Printf.Format(format), value)
    open(markdown_path, "w") do io
        println(io, "# $(study(study_tag).label) selected candidates\n")
        println(io, "| Configuration | Active groups | Global SC sparsity | Global GC sparsity | Validation MSE | Strength | Mask threshold | Test mean(state_Nu) |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows
            println(io, "| $(replace(row.configuration, "<=" => "≤")) | $(active_groups_value(row)) | $(fmt(row.global_sc_sparsity_percent, "%.2f%%")) | $(fmt(row.global_gc_sparsity_percent, "%.2f%%")) | $(fmt(row.validation_mse, "%.4e")) | $(fmt(row.strength, "%.6g")) | $(fmt(row.mask_threshold, "%.6g")) | $(fmt(row.mean_state_nusselt, "%.6f")) |")
        end
        println(io, "\nEach distinct frozen test candidate appears once. If several quality thresholds select the same candidate, all of those thresholds are listed in its Configuration cell. Thresholds without a qualifying pooled-front point contribute no row. Test mean(state_Nu) is the mean over all stored per-step values from the eight 200-step test episodes; lower is better. SC sparsity uses 8×48×3 channel inputs. GC sparsity treats a sensor location as occupied when any of its three channels is active.")
    end
    return (; csv_path, markdown_path)
end

function choose_mask_candidates(configurations, expert_mean, tolerance)
    upper_bound = expert_mean + tolerance * abs(expert_mean)
    selected = Dict{String, Any}()
    for configuration_name in HR_CONFIGURATION_NAMES
        candidates = configurations[configuration_name].candidates
        eligible = filter(
            candidate -> Float64(candidate[:mean_state_nusselt]) <= upper_bound,
            candidates,
        )
        if isempty(eligible)
            best = isempty(candidates) ? missing : minimum(
                Float64(candidate[:mean_state_nusselt]) for candidate in candidates
            )
            error(
                "$configuration_name has no candidate within $(100 * tolerance)% of the " *
                "expert test mean $expert_mean (upper bound $upper_bound; best candidate $best). " *
                "Use --near-expert-relative-tolerance to deliberately change the criterion.",
            )
        end
        selected[configuration_name] = first(sort(eligible; by = frozen -> (
            Int(frozen[:candidate][:active_inputs]),
            Int(frozen[:candidate][:active_groups]),
            Float64(frozen[:mean_state_nusselt]),
            Float64(frozen[:candidate][:validation_matching]),
            string(frozen[:candidate][:candidate_id]),
        )))
    end
    return (; selected, upper_bound)
end

function stripe_matrix(mask)
    values = fill(NaN, 8, STRIPE_COLUMN_COUNT)
    text = fill("", 8, STRIPE_COLUMN_COUNT)
    for vertical in 1:8, horizontal in 1:48, channel in 1:3
        active = Bool(mask[channel, horizontal, vertical])
        start = (horizontal - 1) * STRIPE_SENSOR_WIDTH +
                (channel - 1) * STRIPE_CHANNEL_WIDTH + 1
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

function panel_position(configuration_name)
    method, grouping = split(configuration_name, '-'; limit = 2)
    return (findfirst(==(method), PAPER_METHODS), findfirst(==(grouping), PAPER_GROUPINGS))
end

function mask_panel_titles(mask_selection, expert_mean)
    titles = String[]
    for method in PAPER_METHODS, grouping in PAPER_GROUPINGS
        configuration_name = "$method-$grouping"
        frozen = mask_selection[configuration_name]
        candidate = frozen[:candidate]
        delta = 100 * (Float64(frozen[:mean_state_nusselt]) - expert_mean) / abs(expert_mean)
        push!(titles,
            "$(PAPER_METHOD_NAMES[method]) - $(uppercase(grouping))" *
            "<br><sup>q ≤ $(quality_label(frozen[:quality_thresholds])); " *
            "$(candidate[:active_groups]) groups; ΔNu=$(Printf.format(Printf.Format("%+.2f"), delta))%</sup>",
        )
    end
    return reshape(titles, :, 1)
end

function make_mask_figure(mask_choice, output, study_tag, expert_mean, tolerance)
    selected = mask_choice.selected
    plot = make_subplots(
        rows = 2, cols = 2, vertical_spacing = 0.13, horizontal_spacing = 0.07,
        subplot_titles = mask_panel_titles(selected, expert_mean),
    )
    for configuration_name in HR_CONFIGURATION_NAMES
        row, col = panel_position(configuration_name)
        candidate = selected[configuration_name][:candidate]
        values, text = stripe_matrix(candidate[:global_mask])
        add_trace!(plot, heatmap(
            x = collect(1:STRIPE_COLUMN_COUNT), y = collect(1:8),
            z = values, text = text, zmin = 0, zmax = 3,
            colorscale = STRIPE_COLORSCALE, showscale = false,
            hovertemplate = "%{text}<extra></extra>", xgap = 0, ygap = 1,
        ); row, col)
    end
    for (index, (name, color)) in enumerate(zip(
        ("Inactive", PAPER_CHANNEL_NAMES...),
        (PAPER_INACTIVE_COLOR, PAPER_CHANNEL_COLORS...),
    ))
        add_trace!(plot, scatter(
            x = [NaN], y = [NaN], mode = "markers", name = name,
            marker = attr(
                color = color, size = 11, symbol = "square",
                line = attr(color = "#444444", width = index == 1 ? 1 : 0),
            ),
            legendgroup = "mask_legend", showlegend = true,
        ); row = 1, col = 1)
    end
    layout = Dict{Symbol, Any}(
        :template => "plotly_white", :width => 1450, :height => 900,
        :title => attr(
            text = "$(study(study_tag).label): sparsest test-near-expert masks (≤ $(100 * tolerance)% degradation)",
            x = 0.5, xanchor = "center",
        ),
        :paper_bgcolor => "white", :plot_bgcolor => "white",
        :font => attr(family = "Arial, sans-serif", size = 13, color = "#303030"),
        :margin => attr(l = 80, r = 30, t = 115, b = 100),
        :legend => attr(orientation = "h", x = 0.5, xanchor = "center", y = -0.11, yanchor = "top"),
    )
    for index in 1:4
        layout[axis_key("xaxis", index)] = preserved_axis(plot, axis_key("xaxis", index), attr(
            title = index > 2 ? "Horizontal sensor index" : "",
            range = [0.5, STRIPE_COLUMN_COUNT + 0.5], tickmode = "array",
            tickvals = [stripe_sensor_center(value) for value in 1:4:48],
            ticktext = string.(1:4:48), showgrid = false, zeroline = false,
            showline = true, mirror = true, linecolor = "#3A3A3A",
        ))
        layout[axis_key("yaxis", index)] = preserved_axis(plot, axis_key("yaxis", index), attr(
            title = isodd(index) ? "Vertical sensor index" : "",
            range = [0.5, 8.5], tickmode = "array", tickvals = collect(1:8),
            showgrid = false, zeroline = false, showline = true,
            mirror = true, linecolor = "#3A3A3A",
        ))
    end
    relayout!(plot, layout)
    stem = "figure_1_selected_sensor_masks_$(study_tag)"
    paths = String[]
    for extension in ("svg", "pdf")
        path = joinpath(output, "$stem.$extension")
        PlotlyJS.savefig(plot, path; width = 1450, height = 900)
        push!(paths, path)
    end
    return paths
end

function threshold_styles(configurations)
    thresholds = sort!(unique(
        point.threshold_value
        for data in Base.values(configurations) for point in data.evaluations
    ))
    colors = Dict{Float64, String}()
    ranks = Dict{Float64, Int}()
    positive_index = 0
    for threshold in thresholds
        if iszero(threshold)
            colors[threshold] = PAPER_ZERO_THRESHOLD_COLOR
            ranks[threshold] = 50
        else
            positive_index += 1
            colors[threshold] = PAPER_THRESHOLD_COLORS[mod1(positive_index, length(PAPER_THRESHOLD_COLORS))]
            ranks[threshold] = 100 + positive_index
        end
    end
    return thresholds, colors, ranks
end

function quality_index(value)
    index = findfirst(
        reference -> isapprox(value, reference; atol = 1e-12, rtol = 1e-10),
        HR_QUALITY_THRESHOLDS,
    )
    isnothing(index) && error("Unknown quality threshold $value")
    return index
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

function make_pareto_figure(configurations, output, study_tag)
    thresholds, colors, legend_ranks = threshold_styles(configurations)
    titles = reshape([
        "$(PAPER_METHOD_NAMES[method]) - $(uppercase(grouping))"
        for method in PAPER_METHODS for grouping in PAPER_GROUPINGS
    ], :, 1)
    plot = make_subplots(
        rows = 2, cols = 2, vertical_spacing = 0.13, horizontal_spacing = 0.07,
        subplot_titles = titles,
    )
    shapes = Any[]
    all_losses = Float64[]
    maxima = Dict(grouping => 1 for grouping in PAPER_GROUPINGS)
    shown_mask_thresholds = Set{Float64}()
    for configuration_name in HR_CONFIGURATION_NAMES
        row, col = panel_position(configuration_name)
        index = 2 * (row - 1) + col
        grouping = split(configuration_name, '-'; limit = 2)[2]
        data = configurations[configuration_name]
        eligible = filter(point -> isfinite(point.validation_mse) && point.validation_mse > 0,
                          data.evaluations)
        append!(all_losses, point.validation_mse for point in eligible)
        !isempty(eligible) && (maxima[grouping] = max(
            maxima[grouping], maximum(Int(point.active_groups) for point in eligible)
        ))
        for threshold in thresholds
            points = filter(point -> point.threshold_value == threshold, eligible)
            isempty(points) && continue
            showlegend = !(threshold in shown_mask_thresholds)
            showlegend && push!(shown_mask_thresholds, threshold)
            add_trace!(plot, scattergl(
                x = Int.(getproperty.(points, :active_groups)),
                y = getproperty.(points, :validation_mse),
                mode = "markers", name = "mask τ=$(threshold)",
                showlegend = showlegend, legendgroup = "threshold_$threshold",
                legendrank = legend_ranks[threshold],
                marker = attr(
                    color = colors[threshold], size = 4, opacity = 0.32,
                    symbol = [("circle", "diamond", "square")[Int(point.replicate)] for point in points],
                ),
                customdata = hcat(
                    Int.(getproperty.(points, :active_inputs)),
                    Int.(getproperty.(points, :replicate)),
                    getproperty.(points, :strength),
                    Int.(getproperty.(points, :update)),
                ),
                hovertemplate = "groups=%{x}<br>inputs=%{customdata[0]}<br>MSE=%{y:.4e}<br>replicate=%{customdata[1]}<br>strength=%{customdata[2]:.4g}<br>update=%{customdata[3]}<extra></extra>",
            ); row, col)
        end
        front = sort(filter(point -> point.validation_mse > 0, data.front);
                     by = point -> Int(point.active_inputs))
        add_trace!(plot, scatter(
            x = Int.(getproperty.(front, :active_groups)),
            y = getproperty.(front, :validation_mse),
            mode = "lines+markers", name = "Pooled Pareto front",
            legendgroup = "pooled_front", legendrank = 300, showlegend = index == 1,
            line = attr(color = "#111111", width = 2.2),
            marker = attr(color = "#111111", size = 6, symbol = "circle-open"),
        ); row, col)
        for frozen in data.candidates
            candidate = frozen[:candidate]
            quality_thresholds = sort(Float64.(frozen[:quality_thresholds]); rev = true)
            strictest = minimum(quality_thresholds)
            qindex = quality_index(strictest)
            label = "q≤" * join((@sprintf("%.4g", value) for value in quality_thresholds), "/")
            add_trace!(plot, scatter(
                x = [Int(candidate[:active_groups])],
                y = [Float64(candidate[:validation_matching])],
                mode = "markers+text", text = [label], textposition = "top center",
                textfont = attr(size = 10, color = PAPER_QUALITY_COLORS[qindex]),
                name = "Selected $label", showlegend = false,
                marker = attr(
                    color = PAPER_QUALITY_COLORS[qindex], size = 14,
                    symbol = PAPER_QUALITY_SYMBOLS[qindex],
                    line = attr(color = "#111111", width = 1.0),
                ),
                customdata = [[Int(candidate[:active_inputs]), Float64(frozen[:mean_state_nusselt])]],
                hovertemplate = "groups=%{x}<br>inputs=%{customdata[0]}<br>MSE=%{y:.4e}<br>test mean(state_Nu)=%{customdata[1]:.6f}<extra>$label</extra>",
            ); row, col)
        end
        axis_suffix = index == 1 ? "" : string(index)
        for (qindex, quality_threshold) in enumerate(HR_QUALITY_THRESHOLDS)
            push!(shapes, attr(
                type = "line", xref = "x$axis_suffix domain", x0 = 0, x1 = 1,
                yref = "y$axis_suffix", y0 = quality_threshold, y1 = quality_threshold,
                line = attr(color = PAPER_QUALITY_COLORS[qindex], width = 1.4,
                            dash = ("dash", "dot", "dashdot")[qindex]),
            ))
        end
        if index == 1
            for (qindex, quality_threshold) in enumerate(HR_QUALITY_THRESHOLDS)
                add_trace!(plot, scatter(
                    x = [NaN], y = [NaN], mode = "lines+markers",
                    name = "Quality q≤$(quality_threshold)",
                    legendgroup = "quality_$quality_threshold", legendrank = 400 + qindex,
                    line = attr(color = PAPER_QUALITY_COLORS[qindex], width = 1.4,
                                dash = ("dash", "dot", "dashdot")[qindex]),
                    marker = attr(color = PAPER_QUALITY_COLORS[qindex], size = 10,
                                  symbol = PAPER_QUALITY_SYMBOLS[qindex]),
                ); row, col)
            end
        end
    end
    isempty(all_losses) && error("No finite Higher-Ra evaluation losses were found.")
    y_min = min(minimum(all_losses), minimum(HR_QUALITY_THRESHOLDS))
    y_range = [log10(y_min) - 0.15, log10(10.0)]
    layout = Dict{Symbol, Any}(
        :template => "plotly_white", :width => 1450, :height => 900,
        :title => attr(
            text = "$(study(study_tag).label): evaluation landscapes and pooled Pareto fronts",
            x = 0.5, xanchor = "center",
        ),
        :paper_bgcolor => "white", :plot_bgcolor => "white", :shapes => shapes,
        :font => attr(family = "Arial, sans-serif", size = 13, color = "#303030"),
        :margin => attr(l = 85, r = 30, t = 100, b = 120),
        :legend => attr(orientation = "h", x = 0.5, xanchor = "center", y = -0.13, yanchor = "top"),
    )
    for index in 1:4
        grouping = isodd(index) ? "gc" : "sc"
        layout[axis_key("xaxis", index)] = preserved_axis(plot, axis_key("xaxis", index), attr(
            title = index > 2 ? "Active groups" : "", range = [-0.5, maxima[grouping] + 1],
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
    stem = "figure_s1_pareto_comparison_$(study_tag)"
    svg_path = joinpath(output, "$stem.svg")
    pdf_path = joinpath(output, "$stem.pdf")
    PlotlyJS.savefig(plot, svg_path; width = 1450, height = 900)
    move_glimages_behind_cartesian!(svg_path)
    PlotlyJS.savefig(plot, pdf_path; width = 1450, height = 900)
    return [svg_path, pdf_path]
end

function write_provenance(output, configurations, expert, unactuated, study_tag)
    files = String[expert.path, unactuated.path]
    for data in Base.values(configurations)
        append!(files, [data.paths.status, data.paths.evaluations, data.paths.front, data.paths.selection])
        append!(files, [string(candidate[:test_path]) for candidate in data.candidates])
    end
    sort!(unique!(files))
    path = joinpath(output, "provenance_$(study_tag).sha256")
    open(path, "w") do io
        println(io, "# Higher-Ra paper input manifest for $(study(study_tag).label)")
        println(io, "# Script SHA-256: $(file_sha256(abspath(@__FILE__)))")
        for file in files
            println(io, "$(file_sha256(file))  $(replace(relpath(file, output), '\\' => '/'))")
        end
    end
    return path
end

function build_study_artifacts(options, study_tag)
    configurations = Dict(
        configuration_name => load_configuration(options, study_tag, configuration_name)
        for configuration_name in HR_CONFIGURATION_NAMES
    )
    expert = load_baseline(study_tag, :expert)
    unactuated = load_baseline(study_tag, :unactuated)
    validate_expert_identity(configurations, expert)
    rows = table_rows(configurations, expert, unactuated)
    mask_choice = choose_mask_candidates(
        configurations,
        expert.mean_state_nusselt,
        options.near_expert_relative_tolerance,
    )
    if options.check_only
        candidate_count = sum(length(data.candidates) for data in Base.values(configurations))
        println("$(study(study_tag).label) paper inputs valid: $candidate_count unique frozen candidates across four configurations.")
        return (; study_tag, configurations, expert, unactuated, rows, mask_choice)
    end
    output = output_directory(options, study_tag)
    mkpath(output)
    table = write_table(output, rows, study_tag)
    mask_paths = make_mask_figure(
        mask_choice, output, study_tag, expert.mean_state_nusselt,
        options.near_expert_relative_tolerance,
    )
    pareto_paths = make_pareto_figure(configurations, output, study_tag)
    provenance = write_provenance(output, configurations, expert, unactuated, study_tag)
    metrics_path = atomic_save(
        joinpath(output, "paper_metrics_$(study_tag).jld2");
        schema_version = HR_SCHEMA_VERSION,
        experiment = :higher_ra_paper_artifacts,
        study = study_tag,
        rayleigh = study(study_tag).rayleigh,
        experiment_id = options.experiment_ids[study_tag],
        quality_thresholds = collect(HR_QUALITY_THRESHOLDS),
        near_expert_relative_tolerance = options.near_expert_relative_tolerance,
        near_expert_upper_bound = mask_choice.upper_bound,
        table_rows = rows,
        mask_candidate_ids = Dict(
            configuration_name => string(mask_choice.selected[configuration_name][:candidate][:candidate_id])
            for configuration_name in HR_CONFIGURATION_NAMES
        ),
        table,
        mask_paths,
        pareto_paths,
        provenance,
    )
    println("$(study(study_tag).label) paper artifacts written to $output")
    return (; study_tag, configurations, expert, unactuated, rows, mask_choice,
            output, table, mask_paths, pareto_paths, provenance, metrics_path)
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return nothing
    return [build_study_artifacts(options, study_tag) for study_tag in options.studies]
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
