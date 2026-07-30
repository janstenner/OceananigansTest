include(joinpath(@__DIR__, "MATStabilityExperiment.jl"))
using .MATStabilityExperiment

function usage(io::IO = stdout)
    println(
        io,
        """
        Usage:
          julia --project=. Revision/MAT_Stability/run_worker.jl \\
              --protocol fixed|varying --replicate 1:5 [options]

        Options:
          --episodes N       Override the protocol default (2000 fixed, 4000 varying).
          --dry-run          Build and verify all configs without training episodes.
          --overwrite        Replace already complete matching result files.
          --results-dir DIR  Override Revision/MAT_Stability/results.
          --help             Show this message.
        """,
    )
end

function parse_arguments(arguments)
    options = Dict{String, Any}(
        "protocol" => nothing,
        "replicate" => nothing,
        "episodes" => nothing,
        "dry_run" => false,
        "overwrite" => false,
        "results_dir" => nothing,
    )

    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--dry-run"
            options["dry_run"] = true
        elseif argument == "--overwrite"
            options["overwrite"] = true
        elseif argument in ("--protocol", "--replicate", "--episodes", "--results-dir")
            index == length(arguments) && error("Missing value after $argument.")
            value = arguments[index + 1]
            key = replace(argument[3:end], "-" => "_")
            options[key] = value
            index += 1
        else
            error("Unknown argument '$argument'. Use --help for usage.")
        end
        index += 1
    end

    isnothing(options["protocol"]) && error("--protocol is required.")
    isnothing(options["replicate"]) && error("--replicate is required.")
    options["replicate"] = parse(Int, options["replicate"])
    if !isnothing(options["episodes"])
        options["episodes"] = parse(Int, options["episodes"])
    end
    return options
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return

    keywords = (
        episodes = options["episodes"],
        dry_run = options["dry_run"],
        overwrite = options["overwrite"],
    )
    result_directory = options["results_dir"]
    output = if isnothing(result_directory)
        run_worker(options["protocol"], options["replicate"]; keywords...)
    else
        run_worker(
            options["protocol"],
            options["replicate"];
            keywords...,
            results_directory = abspath(result_directory),
        )
    end
    println("Worker completed: $output")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
