ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
include(joinpath(@__DIR__, "..", "VaryingIC_Corpus", "VaryingICCorpus.jl"))
include(joinpath(@__DIR__, "DistillationCorpus.jl"))

function prepare_usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --project=. Revision/Expert_Apprentice_Distillation/prepare_corpus_workers.jl \\
        --jobs-file PATH [options]

    Options:
      --protocol all|fixed|varying   Default: varying.
      --split all|train|validation|test
                                    Default: train.
      --worker-dir PATH              Worker-result root.
      --fixed-expert-path PATH       Expected Fixed-IC expert.
      --varying-expert-path PATH     Expected Varying-IC expert.
      --run-seed N                   Fresh-MAT smoke-test seed.
      --allow-fresh-expert           Smoke tests only.
      --overwrite                    Include already complete jobs.
      --help                         Show this message.
    """)
end

function parse_prepare_arguments(arguments)
    options = Dict{String, Any}(
        "protocol" => "varying",
        "split" => "train",
        "worker_dir" => DISTILLATION_WORKER_DIRECTORY,
        "jobs_file" => nothing,
        "fixed_expert_path" => nothing,
        "varying_expert_path" => nothing,
        "run_seed" => 600600,
        "allow_fresh_expert" => false,
        "overwrite" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            prepare_usage()
            return nothing
        elseif argument == "--overwrite"
            options["overwrite"] = true
        elseif argument == "--allow-fresh-expert"
            options["allow_fresh_expert"] = true
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
    lowercase(string(options["protocol"])) in ("all", "fixed", "varying") || error(
        "--protocol must be all, fixed, or varying.",
    )
    lowercase(string(options["split"])) in ("all", "train", "validation", "test") || error(
        "--split must be all, train, validation, or test.",
    )
    options["run_seed"] = options["run_seed"] isa Integer ?
        options["run_seed"] : parse(Int, options["run_seed"])
    return options
end

function expected_expert_identifier(
    protocol;
    explicit_path = nothing,
    allow_fresh_expert::Bool = false,
    run_seed::Integer = 600600,
)
    path = find_distillation_expert(protocol; explicit_path)
    !isnothing(path) && return "sha256:$(file_sha256(path))"
    allow_fresh_expert && return "fresh-initialized-mat-seed-$run_seed"
    return nothing
end

function matching_worker_complete(
    path::AbstractString;
    protocol,
    expected_identifier,
)
    isfile(path) || return false
    isnothing(expected_identifier) && return false
    try
        result = load_distillation_worker(path)
        metadata = distillation_value(result, :expert_metadata)
        identifier = string(get(metadata, :identifier, get(metadata, "identifier", "unknown")))
        expected_offsets = protocol === :fixed ? [nothing] : collect(DISTILLATION_OFFSETS)
        return distillation_value(result, :complete) === true &&
               Int(distillation_value(result, :rollout_steps)) == DISTILLATION_ROLLOUT_STEPS &&
               collect(distillation_value(result, :offsets)) == expected_offsets &&
               identifier == expected_identifier
    catch
        return false
    end
end

function planned_jobs(
    protocol_selection,
    split_selection;
    worker_directory,
    overwrite::Bool,
    fixed_expert_path = nothing,
    varying_expert_path = nothing,
    allow_fresh_expert::Bool = false,
    run_seed::Integer = 600600,
)
    protocols = protocol_selection == "all" ? (:fixed, :varying) : (Symbol(protocol_selection),)
    splits = split_selection == "all" ? DISTILLATION_SPLITS : (Symbol(split_selection),)
    jobs = NamedTuple[]

    fixed_identifier = :fixed in protocols ? expected_expert_identifier(
        :fixed;
        explicit_path = fixed_expert_path,
        allow_fresh_expert,
        run_seed,
    ) : nothing
    varying_identifier = :varying in protocols ? expected_expert_identifier(
        :varying;
        explicit_path = varying_expert_path,
        allow_fresh_expert,
        run_seed,
    ) : nothing

    if :fixed in protocols
        path = distillation_worker_path(:fixed; worker_directory)
        if overwrite || !matching_worker_complete(
            path;
            protocol = :fixed,
            expected_identifier = fixed_identifier,
        )
            push!(jobs, (
                protocol = :fixed,
                split = :shared,
                base_seed = "-",
                mirror = "-",
                output_path = path,
            ))
        end
    end

    if :varying in protocols
        for split in splits
            seeds = sort!(collect(keys(CORPUS[split])))
            isempty(seeds) && error("Varying-IC split :$split contains no basis snapshots.")
            for seed in seeds, mirror in (false, true)
                path = distillation_worker_path(
                    :varying,
                    split;
                    base_seed = seed,
                    mirror,
                    worker_directory,
                )
                if overwrite || !matching_worker_complete(
                    path;
                    protocol = :varying,
                    expected_identifier = varying_identifier,
                )
                    push!(jobs, (
                        protocol = :varying,
                        split,
                        base_seed = string(seed),
                        mirror = string(mirror),
                        output_path = path,
                    ))
                end
            end
        end
    end
    return jobs
end

function prepare_main(arguments = ARGS)
    options = parse_prepare_arguments(arguments)
    isnothing(options) && return
    jobs = planned_jobs(
        lowercase(string(options["protocol"])),
        lowercase(string(options["split"]));
        worker_directory = abspath(string(options["worker_dir"])),
        overwrite = options["overwrite"],
        fixed_expert_path = options["fixed_expert_path"],
        varying_expert_path = options["varying_expert_path"],
        allow_fresh_expert = options["allow_fresh_expert"],
        run_seed = options["run_seed"],
    )
    jobs_file = abspath(string(options["jobs_file"]))
    mkpath(dirname(jobs_file))
    open(jobs_file, "w") do io
        println(io, "protocol\tsplit\tbase_seed\tmirror\toutput_path")
        for job in jobs
            println(
                io,
                join((job.protocol, job.split, job.base_seed, job.mirror, job.output_path), '\t'),
            )
        end
    end
    println("Prepared $(length(jobs)) missing distillation worker jobs in $jobs_file")
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        prepare_main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
