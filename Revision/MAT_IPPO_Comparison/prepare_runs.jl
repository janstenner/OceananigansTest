include(joinpath(@__DIR__, "MATIPPOExperiment.jl"))
using .MATIPPOExperiment

function parse_bool(value)
    normalized = lowercase(value)
    normalized in ("true", "1", "yes") && return true
    normalized in ("false", "0", "no") && return false
    error("Expected true or false, got '$value'.")
end

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --project=. Revision/MAT_IPPO_Comparison/prepare_runs.jl [options]

    Required:
      --n-runs N                     Number of seed pairs in this launch batch.
      --look-for-imports true|false  Reuse complete Package-3 pairs or generate seeds.
      --jobs-file PATH               Write the worker manifest as tab-separated text.

    Options:
      --protocol all|fixed|varying   Default: all.
      --results-dir PATH             Default: MAT_IPPO_Comparison/results.
      --package3-dir PATH            Default: MAT_Stability/results.
      --preview                      Inspect without persisting plan/imports.
      --overwrite                    Re-run the most recent matching seed pairs.
      --help                         Show this message.

    Underscore spellings such as --n_runs are accepted too.
    """)
end

function parse_arguments(arguments)
    options = Dict{String, Any}(
        "n_runs" => nothing,
        "look_for_imports" => nothing,
        "jobs_file" => nothing,
        "protocol" => "all",
        "results_dir" => MATIPPOExperiment.DEFAULT_RESULTS_DIRECTORY,
        "package3_dir" => joinpath(@__DIR__, "..", "MAT_Stability", "results"),
        "preview" => false,
        "overwrite" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--preview"
            options["preview"] = true
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
    for key in ("n_runs", "look_for_imports", "jobs_file")
        isnothing(options[key]) && error("--$(replace(key, "_" => "-")) is required.")
    end
    options["n_runs"] = parse(Int, options["n_runs"])
    options["look_for_imports"] = parse_bool(options["look_for_imports"])
    return options
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return
    prepared = prepare_jobs(;
        n_runs = options["n_runs"],
        look_for_imports = options["look_for_imports"],
        protocol = options["protocol"],
        results_directory = abspath(options["results_dir"]),
        package3_directory = abspath(options["package3_dir"]),
        preview = options["preview"],
        overwrite = options["overwrite"],
    )

    jobs_file = abspath(options["jobs_file"])
    mkpath(dirname(jobs_file))
    open(jobs_file, "w") do io
        println(io, "task\trun_id\tprotocol\talgorithm\tresult_path")
        for job in prepared.jobs
            println(
                io,
                join((job.task, job.run_id, job.protocol, job.algorithm, job.path), '\t'),
            )
        end
    end
    println("Selected seed pairs: $(length(prepared.entries))")
    for entry in prepared.entries
        println(
            "  $(entry.run_id): run_seed=$(entry.run_seed), " *
            "ic_seed=$(entry.ic_seed), origin=$(entry.origin)",
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
