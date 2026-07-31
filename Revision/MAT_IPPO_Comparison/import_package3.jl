include(joinpath(@__DIR__, "MATIPPOExperiment.jl"))
using .MATIPPOExperiment

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --project=. Revision/MAT_IPPO_Comparison/import_package3.jl --n-runs N [options]

    Options:
      --results-dir PATH   Package-4 results directory.
      --package3-dir PATH  Package-3 MAT stability results directory.
      --preview            Validate and list candidates without copying or changing the plan.
      --help               Show this message.
    """)
end

function main(arguments = ARGS)
    count = nothing
    results_directory = MATIPPOExperiment.DEFAULT_RESULTS_DIRECTORY
    package3_directory = joinpath(@__DIR__, "..", "MAT_Stability", "results")
    preview = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return
        elseif argument == "--preview"
            preview = true
        elseif argument in ("--n-runs", "--n_runs", "--results-dir", "--results_dir",
                            "--package3-dir", "--package3_dir")
            index == length(arguments) && error("Missing value after $argument.")
            value = arguments[index + 1]
            if argument in ("--n-runs", "--n_runs")
                count = parse(Int, value)
            elseif argument in ("--results-dir", "--results_dir")
                results_directory = abspath(value)
            else
                package3_directory = abspath(value)
            end
            index += 1
        else
            error("Unknown argument '$argument'.")
        end
        index += 1
    end
    isnothing(count) && error("--n-runs is required.")
    entries = import_package3!(;
        count,
        results_directory,
        package3_directory,
        preview,
    )
    println(preview ? "Import preview:" : "Imported/selected entries:")
    for entry in entries
        println("  $(entry.run_id): run_seed=$(entry.run_seed), ic_seed=$(entry.ic_seed)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end

