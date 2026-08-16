include(joinpath(@__DIR__, "MATStabilityExperiment.jl"))
using .MATStabilityExperiment

function usage(io::IO = stdout)
    println(
        io,
        """
        Usage:
          julia --project=. Revision/MAT_Stability/prepare_runs.jl \\
              --jobs-file PATH [options]

        Options:
          --protocol all|fixed|varying  Default: all.
          --results-dir PATH            Default: Revision/MAT_Stability/results.
          --dry-run                     Plan zero-episode verification workers.
          --preview                     Inspect without persisting the frozen run plan.
          --overwrite                   Include already complete matching results.
          --help                        Show this message.
        """,
    )
end

function parse_arguments(arguments)
    options = Dict{String, Any}(
        "jobs_file" => nothing,
        "protocol" => "all",
        "results_dir" => MATStabilityExperiment.DEFAULT_RESULTS_DIRECTORY,
        "dry_run" => false,
        "preview" => false,
        "overwrite" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument in ("--dry-run", "--preview", "--overwrite")
            options[replace(argument[3:end], "-" => "_")] = true
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
    isnothing(options["jobs_file"]) && error("--jobs-file is required.")
    return options
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return
    prepared = prepare_jobs(;
        protocol = options["protocol"],
        results_directory = abspath(options["results_dir"]),
        dry_run = options["dry_run"],
        preview = options["preview"],
        overwrite = options["overwrite"],
    )

    jobs_file = abspath(options["jobs_file"])
    mkpath(dirname(jobs_file))
    open(jobs_file, "w") do io
        println(
            io,
            "protocol\treplicate\tconfig\trun_seed\tic_seed\tepisode_target\tresult_path",
        )
        for job in prepared.jobs
            println(
                io,
                join(
                    (
                        job.protocol,
                        job.replicate,
                        job.config_name,
                        job.run_seed,
                        job.ic_seed,
                        job.episode_target,
                        job.path,
                    ),
                    '\t',
                ),
            )
        end
    end

    println("Frozen replicate seed pairs:")
    for entry in prepared.plan["entries"]
        println(
            "  replicate $(entry.replicate): run_seed=$(entry.run_seed), " *
            "ic_seed=$(entry.ic_seed)",
        )
    end
    println("Pending jobs: $(length(prepared.jobs))")
    println("Worker manifest: $jobs_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
