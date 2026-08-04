ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
include(joinpath(@__DIR__, "DistillationCorpus.jl"))

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --project=. Revision/Expert_Apprentice_Distillation/run_corpus_worker.jl \\
        --protocol fixed|varying [options]

    Varying-IC options:
      --split train|validation|test
      --base-seed SEED
      --mirror true|false

    Common options:
      --expert-path PATH       Explicit expert checkpoint.
      --worker-dir PATH        Override the worker-result root.
      --steps N                Rollout length; production default: 200.
      --offset-start N         Testing/sharding override; default: 0.
      --offset-count N         Testing override; production default: 96.
      --run-seed N             MAT initialization seed; default: 600600.
      --allow-fresh-expert     Smoke tests only; use the freshly initialized MAT.
      --overwrite              Replace a matching complete worker file.
      --help                   Show this message.
    """)
end

function parse_worker_arguments(arguments)
    options = Dict{String, Any}(
        "protocol" => nothing,
        "split" => "train",
        "base_seed" => nothing,
        "mirror" => nothing,
        "expert_path" => nothing,
        "worker_dir" => joinpath(@__DIR__, "worker_results"),
        "steps" => 200,
        "offset_start" => 0,
        "offset_count" => 96,
        "run_seed" => 600600,
        "allow_fresh_expert" => false,
        "overwrite" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--allow-fresh-expert"
            options["allow_fresh_expert"] = true
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

    isnothing(options["protocol"]) && error("--protocol is required.")
    for key in ("steps", "offset_start", "offset_count", "run_seed")
        options[key] = options[key] isa Integer ? options[key] : parse(Int, options[key])
    end
    options["steps"] > 0 || error("--steps must be positive.")
    0 <= options["offset_start"] <= 95 || error("--offset-start must be in 0:95.")
    1 <= options["offset_count"] <= 96 || error("--offset-count must be in 1:96.")
    options["offset_start"] + options["offset_count"] <= 96 || error(
        "--offset-start + --offset-count must not exceed 96.",
    )

    protocol = Symbol(lowercase(string(options["protocol"])))
    protocol in (:fixed, :varying) || error("--protocol must be fixed or varying.")
    options["protocol"] = protocol
    if protocol === :varying
        isnothing(options["base_seed"]) && error("--base-seed is required for varying IC.")
        isnothing(options["mirror"]) && error("--mirror is required for varying IC.")
        options["base_seed"] = parse(Int, string(options["base_seed"]))
        mirror_string = lowercase(string(options["mirror"]))
        mirror_string in ("true", "false") || error("--mirror must be true or false.")
        options["mirror"] = mirror_string == "true"
    end
    return options
end

function include_revision_mat(protocol::Symbol, runtime_directory::AbstractString, run_seed::Integer)
    ENV["REVISION_RUN_SEED"] = string(run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    run_file = protocol === :fixed ? "FixedIC_MAT.jl" : "VaryingIC_MAT.jl"
    Base.include(Main, joinpath(@__DIR__, "..", "Run_Files", run_file))
    return nothing
end

function main(arguments = ARGS)
    options = parse_worker_arguments(arguments)
    isnothing(options) && return

    protocol = options["protocol"]
    split = Symbol(lowercase(string(options["split"])))
    base_seed = options["base_seed"]
    mirror = something(options["mirror"], false)
    worker_directory = abspath(string(options["worker_dir"]))
    worker_label = protocol === :fixed ? "fixed_shared" :
        "$(split)_$(base_seed)_mirror_$(mirror ? 1 : 0)"
    runtime_directory = joinpath(worker_directory, "runtime", worker_label)

    include_revision_mat(protocol, runtime_directory, options["run_seed"])
    Base.include(Main, joinpath(@__DIR__, "execute_corpus_worker.jl"))
    executor = Core.eval(Main, :execute_loaded_corpus_worker)
    Base.invokelatest(executor, options)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
