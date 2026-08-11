include(joinpath(@__DIR__, "run_study_test_worker.jl"))

function parse_episode_arguments(arguments)
    values = Dict(
        "protocol" => nothing,
        "results_dir" => joinpath(@__DIR__, "results", "study"),
        "manifest" => nothing,
        "controller_index" => nothing,
        "case_index" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        startswith(argument, "--") || error("Unknown argument '$argument'.")
        index == length(arguments) && error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    isnothing(values["protocol"]) && error("--protocol is required.")
    isnothing(values["controller_index"]) && error("--controller-index is required.")
    isnothing(values["case_index"]) && error("--case-index is required.")
    protocol = normalize_protocol(values["protocol"])
    results_root = abspath(string(values["results_dir"]))
    manifest = isnothing(values["manifest"]) ?
        joinpath(results_root, string(protocol), "analysis", "candidate_manifest.jld2") :
        abspath(string(values["manifest"]))
    return (
        options = (; protocol, results_root, manifest, parallel_test = true),
        controller_index = parse(Int, string(values["controller_index"])),
        case_index = parse(Int, string(values["case_index"])),
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        parsed = parse_episode_arguments(ARGS)
        run_single_test_episode(parsed.options, parsed.controller_index, parsed.case_index)
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
