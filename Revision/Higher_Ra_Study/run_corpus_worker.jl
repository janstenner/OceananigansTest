ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"

include(joinpath(@__DIR__, "HigherRaDistillationCorpus.jl"))

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/Higher_Ra_Study/run_corpus_worker.jl \\
        --study ra5e4|ra1e5 --split train|validation|test \\
        --base-seed N --mirror true|false [options]

    Options:
      --expert-path PATH       Matching compact Higher-Ra expert.
      --worker-dir PATH        Study-specific worker-result root.
      --steps N                Rollout length; production default: 200.
      --run-seed N             Runtime initialization seed; default: 600600.
      --overwrite              Replace a matching complete shard.
      --help
    """)
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "study" => nothing,
        "split" => nothing,
        "base_seed" => nothing,
        "mirror" => nothing,
        "expert_path" => nothing,
        "worker_dir" => nothing,
        "steps" => DISTILLATION_ROLLOUT_STEPS,
        "run_seed" => 600600,
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
    for required in ("study", "split", "base_seed", "mirror")
        isnothing(options[required]) && error("--$(replace(required, "_" => "-")) is required.")
    end
    study = higher_ra_study(options["study"])
    split = normalize_distillation_split(options["split"])
    mirror_text = lowercase(string(options["mirror"]))
    mirror_text in ("true", "false") || error("--mirror must be true or false.")
    steps = options["steps"] isa Integer ? Int(options["steps"]) : parse(Int, string(options["steps"]))
    steps > 0 || error("--steps must be positive.")
    run_seed = options["run_seed"] isa Integer ? Int(options["run_seed"]) :
        parse(Int, string(options["run_seed"]))
    run_seed >= 0 || error("--run-seed must be nonnegative.")
    return Dict{String, Any}(
        "study" => study,
        "protocol" => :varying,
        "split" => split,
        "base_seed" => parse(Int, string(options["base_seed"])),
        "mirror" => mirror_text == "true",
        "expert_path" => abspath(isnothing(options["expert_path"]) ?
            study.expert : string(options["expert_path"])),
        "worker_dir" => abspath(isnothing(options["worker_dir"]) ?
            higher_ra_worker_root(study.tag) : string(options["worker_dir"])),
        "steps" => steps,
        "run_seed" => run_seed,
        "overwrite" => Bool(options["overwrite"]),
    )
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    study = options["study"]
    validate_higher_ra_sources(study; expert_path = options["expert_path"])
    worker_label = "$(study.tag)_$(options["split"])_$(options["base_seed"])_mirror_$(options["mirror"] ? 1 : 0)"
    runtime_directory = joinpath(options["worker_dir"], "runtime", worker_label)
    ENV["REVISION_RUN_SEED"] = string(options["run_seed"])
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    Base.include(Main, study.run_file)
    Base.include(Main, joinpath(@__DIR__, "execute_corpus_worker.jl"))
    executor = Core.eval(Main, :execute_loaded_higher_ra_corpus_worker)
    return Base.invokelatest(executor, options)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end

