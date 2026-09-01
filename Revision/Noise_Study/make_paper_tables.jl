using Dates
using JLD2
using Printf
using SHA
using Statistics

include(joinpath(@__DIR__, "NoiseStudy.jl"))
using .NoiseStudy

const DEFAULT_NOISE_RESULTS = joinpath(@__DIR__, "results")
const PAPER_PROTOCOLS = (:fixed, :varying)
const PAPER_CONTROLLERS = (:expert, :c_match, :sparse)
const PAPER_CONTROLLER_LABELS = Dict(
    :expert => "Dense expert",
    :sparse => "Sparse apprentice",
    :c_match => "C_match apprentice",
)

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/Noise_Study/make_paper_tables.jl \\
        [--experiment-id ID] [--results-dir PATH] [--output-dir PATH] [--check-only]

    Without --experiment-id, the script independently selects the newest
    Noise-Study experiment directory available for Fixed and Varying IC.
    Different or relocated experiment IDs produce warnings but do not abort.
    Missing worker results are written as NA so a partial table can be created
    while the other protocol is still running.
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

file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function write_provenance(output, source_files, experiments)
    path = joinpath(output, "provenance.sha256")
    files = sort!(unique!(abspath.(source_files)))
    open(path, "w") do io
        println(io, "# Package-10 paper table input manifest")
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
    if options.check_only
        println("Noise-Study paper-table inputs read: $(length(long_rows)) complete worker results, $(length(missing_results)) missing worker results.")
        return (; options, rows, long_rows, source_files, missing_results)
    end
    mkpath(options.output)
    csv_path = write_csv(options.output, rows)
    markdown_path = write_markdown(options.output, rows, options.experiments)
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
        provenance_path,
        created_at = string(Dates.now(Dates.UTC)),
    )
    println("Package-10 paper table written to $(options.output)")
    println("  CSV: $csv_path")
    println("  Markdown: $markdown_path")
    isempty(missing_results) || println("  Missing worker results represented as NA: $(length(missing_results))")
    return (; options, rows, long_rows, csv_path, markdown_path, provenance_path, metrics_path)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
