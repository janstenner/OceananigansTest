ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"

include(joinpath(@__DIR__, "HigherRaDistillationCorpus.jl"))

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/Higher_Ra_Study/prepare_corpus_workers.jl \\
        --study ra5e4|ra1e5 --jobs-file PATH [options]

    Options:
      --split all|train|validation|test  Default: train.
      --worker-dir PATH                  Study-specific worker-result root.
      --expert-path PATH                 Matching compact Higher-Ra expert.
      --jobs-file PATH                   Required TSV output.
      --overwrite                        Include matching complete shards.
      --help
    """)
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "study" => nothing,
        "split" => "train",
        "worker_dir" => nothing,
        "expert_path" => nothing,
        "jobs_file" => nothing,
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
    isnothing(options["study"]) && error("--study is required.")
    isnothing(options["jobs_file"]) && error("--jobs-file is required.")
    study = higher_ra_study(options["study"])
    split = Symbol(lowercase(string(options["split"])))
    split in (:all, DISTILLATION_SPLITS...) || error(
        "--split must be all, train, validation, or test.",
    )
    return (;
        study,
        split,
        worker_directory = abspath(isnothing(options["worker_dir"]) ?
            higher_ra_worker_root(study.tag) : string(options["worker_dir"])),
        expert_path = abspath(isnothing(options["expert_path"]) ?
            study.expert : string(options["expert_path"])),
        jobs_file = abspath(string(options["jobs_file"])),
        overwrite = Bool(options["overwrite"]),
    )
end

function planned_jobs(options)
    source = validate_higher_ra_sources(options.study; expert_path = options.expert_path)
    corpus_seeds = load_higher_ra_corpus_plan(options.study)
    splits = options.split === :all ? DISTILLATION_SPLITS : (options.split,)
    jobs = NamedTuple[]
    for split in splits
        for base_seed in corpus_seeds[split], mirror in (false, true)
            path = distillation_worker_path(
                :varying,
                split;
                base_seed,
                mirror,
                worker_directory = options.worker_directory,
            )
            matching = higher_ra_worker_matches(
                path,
                options.study,
                split,
                source.expert_sha256,
                source.state_corpus_sha256,
                source.run_file_sha256,
            )
            if options.overwrite || !matching
                push!(jobs, (;
                    study = options.study.tag,
                    split,
                    base_seed,
                    mirror,
                    output_path = abspath(path),
                ))
            end
        end
    end
    return jobs
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    jobs = planned_jobs(options)
    mkpath(dirname(options.jobs_file))
    open(options.jobs_file, "w") do io
        println(io, "study\tsplit\tbase_seed\tmirror\toutput_path")
        for job in jobs
            println(io, join((job.study, job.split, job.base_seed, job.mirror, job.output_path), '\t'))
        end
    end
    println("Prepared $(length(jobs)) missing $(options.study.label) distillation jobs in $(options.jobs_file)")
    return options.jobs_file
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()

