include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining

function usage()
    println(
        """
        Usage: julia --project=. Revision/MAT_expert_training/run_training_worker.jl \\
          --protocol fixed|varying --run-id RUN_ID [options]

        Options:
          --results-dir PATH         Expert-training result root.
          --source-results-dir PATH  MAT-IPPO Package-4 result root.
          --help                     Show this message.
        """,
    )
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "protocol" => nothing,
        "run_id" => nothing,
        "results_dir" => MATExpertTraining.DEFAULT_RESULTS_DIRECTORY,
        "source_results_dir" => MATExpertTraining.DEFAULT_SOURCE_RESULTS_DIRECTORY,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            exit(0)
        elseif argument in ("--protocol", "--run-id", "--run_id", "--results-dir",
                            "--results_dir", "--source-results-dir", "--source_results_dir")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], '-' => '_')
            options[key] = arguments[index + 1]
            index += 2
            continue
        else
            error("Unknown argument: $argument")
        end
    end
    isnothing(options["protocol"]) && error("--protocol is required.")
    isnothing(options["run_id"]) && error("--run-id is required.")
    return options
end

options = parse_options(ARGS)
MATExpertTraining.run_training_worker(
    protocol = options["protocol"],
    run_id = options["run_id"],
    results_directory = options["results_dir"],
    source_results_directory = options["source_results_dir"],
)

