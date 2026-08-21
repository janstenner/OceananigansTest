include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining

function usage()
    println(
        """
        Usage: julia --project=. Revision/MAT_expert_training/finalize_best_so_far.jl [options]

        Gracefully stops training with the current global best-so-far snapshot.
        Active workers finish their current episode and save final checkpoints;
        the waiting test/export worker then evaluates the frozen best snapshot.

        Options:
          --protocol fixed|varying|all  Protocol to finalize; default: all.
          --results-dir PATH            Expert-training result root.
          --help                        Show this message.
        """,
    )
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "protocol" => "all",
        "results_dir" => MATExpertTraining.DEFAULT_RESULTS_DIRECTORY,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            exit(0)
        elseif argument in ("--protocol", "--results-dir", "--results_dir")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], '-' => '_')
            options[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument: $argument")
        end
    end
    return options
end

options = parse_options(ARGS)
MATExpertTraining.finalize_best_so_far!(
    results_directory = string(options["results_dir"]),
    protocol = options["protocol"],
)
