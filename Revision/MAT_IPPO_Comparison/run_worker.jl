include(joinpath(@__DIR__, "MATIPPOExperiment.jl"))
using .MATIPPOExperiment

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --project=. Revision/MAT_IPPO_Comparison/run_worker.jl \\
        --task train|validate --run-id ID --protocol fixed|varying \\
        --algorithm mat|ippo [options]

    Options:
      --results-dir PATH  Override MAT_IPPO_Comparison/results.
      --episodes N        Testing override; normal runs use 2000/4000.
      --overwrite         Replace a matching result and its validation.
      --help              Show this message.
    """)
end

function parse_arguments(arguments)
    options = Dict{String, Any}(
        "task" => "train",
        "run_id" => nothing,
        "protocol" => nothing,
        "algorithm" => nothing,
        "results_dir" => MATIPPOExperiment.DEFAULT_RESULTS_DIRECTORY,
        "episodes" => nothing,
        "overwrite" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--overwrite"
            options["overwrite"] = true
        elseif startswith(argument, "--")
            index == length(arguments) && error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(options, key) || error("Unknown option '$argument'.")
            options[key] = arguments[index + 1]
            index += 1
        else
            error("Unknown argument '$argument'.")
        end
        index += 1
    end
    for key in ("run_id", "protocol", "algorithm")
        isnothing(options[key]) && error("--$(replace(key, "_" => "-")) is required.")
    end
    !isnothing(options["episodes"]) && (options["episodes"] = parse(Int, options["episodes"]))
    return options
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return
    output = run_worker(;
        task = options["task"],
        run_id = options["run_id"],
        protocol = options["protocol"],
        algorithm = options["algorithm"],
        results_directory = abspath(options["results_dir"]),
        episodes = options["episodes"],
        overwrite = options["overwrite"],
    )
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

