include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining

function main(arguments)
    preview = "--preview" in arguments
    results_directory = MATExpertTraining.DEFAULT_RESULTS_DIRECTORY
    source_results_directory = MATExpertTraining.DEFAULT_SOURCE_RESULTS_DIRECTORY

    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--preview"
            index += 1
        elseif argument in ("--results-dir", "--results_dir")
            index < length(arguments) || error("Missing value after $argument.")
            results_directory = arguments[index + 1]
            index += 2
        elseif argument in ("--source-results-dir", "--source_results_dir")
            index < length(arguments) || error("Missing value after $argument.")
            source_results_directory = arguments[index + 1]
            index += 2
        else
            error("Unknown argument: $argument")
        end
    end

    records = MATExpertTraining.freeze_selection_manifest!(;
        results_directory,
        source_results_directory,
        preview,
    )
    println(preview ? "Validated preview candidate selection:" : "Frozen candidate selection:")
    for candidate in records
        println(
            "  $(candidate.protocol) rank $(candidate.rank): $(candidate.run_id), " *
            "frozen_validation=$(candidate.validation_score), " *
            "observed_validation=$(candidate.observed_validation_score), " *
            "delta=$(candidate.validation_score_delta), " *
            "checkpoint_sha256=$(candidate.checkpoint_sha256)",
        )
    end
    println("Fixed stop: completed episode reward > $(MATExpertTraining.FIXED_THRESHOLD)")
    println("Varying stop: mean of latest 100 completed episode rewards > $(MATExpertTraining.VARYING_THRESHOLD)")
end

main(ARGS)
