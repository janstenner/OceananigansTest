using Dates
using JLD2
using SHA
using Statistics
using UUIDs

const UNACTUATED_SCHEMA_VERSION = 1
const UNACTUATED_STEPS = 200
const UNACTUATED_RUN_SEED = 20_260_902
const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_BASELINE_ROOT = joinpath(@__DIR__, "Baselines")

const UNACTUATED_STUDIES = (
    ra5e4 = (
        tag = :ra5e4,
        label = "Ra=5e4",
        protocol = :varying_ra5e4,
        rayleigh = 5.0e4,
        run_file = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "VaryingIC_MAT_Ra5e4.jl"),
        corpus_file = joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus_Ra5e4.jld2"),
    ),
    ra1e5 = (
        tag = :ra1e5,
        label = "Ra=1e5",
        protocol = :varying_ra1e5,
        rayleigh = 1.0e5,
        run_file = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "VaryingIC_MAT_Ra1e5.jl"),
        corpus_file = joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus_Ra1e5.jld2"),
    ),
)

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. \
        Revision/Higher_Ra_Study/compute_unactuated_baselines.jl [options]

    Options:
      --study all|ra5e4|ra1e5  Study selection (default: all).
      --results-dir PATH        Output root (default: Higher_Ra_Study/Baselines).
      --run-seed N              Runtime initialization seed (default: 20260902).
      --overwrite               Recompute matching complete baselines.
      --help

    Each selected study evaluates zero actuator actions on its eight Higher-Ra
    Varying-IC test cases for 200 control steps. Results are written to
    Baselines/<ra>/unactuated.jld2.
    """)
end

function parse_options(arguments)
    values = Dict{String, Any}(
        "study" => "all",
        "results_dir" => DEFAULT_BASELINE_ROOT,
        "run_seed" => UNACTUATED_RUN_SEED,
        "overwrite" => false,
        "run_one" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--overwrite"
            values["overwrite"] = true
            index += 1
        elseif startswith(argument, "--")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(values, key) || error("Unknown option '$argument'.")
            values[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument '$argument'.")
        end
    end

    selected_study = Symbol(lowercase(string(values["study"])))
    selected_study in (:all, keys(UNACTUATED_STUDIES)...) || error(
        "--study must be all, ra5e4, or ra1e5.",
    )
    run_one = isnothing(values["run_one"]) ? nothing :
        Symbol(lowercase(string(values["run_one"])))
    isnothing(run_one) || run_one in keys(UNACTUATED_STUDIES) || error(
        "--run-one must be ra5e4 or ra1e5.",
    )
    run_seed = values["run_seed"] isa Integer ? Int(values["run_seed"]) :
        parse(Int, string(values["run_seed"]))
    run_seed >= 0 || error("--run-seed must be nonnegative.")

    return (;
        selected_study,
        results_root = abspath(string(values["results_dir"])),
        run_seed,
        overwrite = Bool(values["overwrite"]),
        run_one,
    )
end

file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

latest_binding(name::Symbol) = Base.invokelatest(() -> getfield(@__MODULE__, name))

function dict_value(mapping, key::Symbol)
    haskey(mapping, key) && return mapping[key]
    haskey(mapping, string(key)) && return mapping[string(key)]
    error("Missing '$key'.")
end

function study_sources(study)
    isfile(study.run_file) || error("Run file is missing: $(study.run_file)")
    isfile(study.corpus_file) || error("Test corpus is missing: $(study.corpus_file)")
    return (;
        run_file_path = abspath(study.run_file),
        run_file_sha256 = file_sha256(study.run_file),
        test_data_path = abspath(study.corpus_file),
        test_data_sha256 = file_sha256(study.corpus_file),
    )
end

baseline_path(options, study) = joinpath(
    options.results_root,
    string(study.tag),
    "unactuated.jld2",
)

function baseline_is_current(path, options, study, sources)
    isfile(path) || return false
    return try
        JLD2.jldopen(path, "r") do file
            Int(read(file, "schema_version")) == UNACTUATED_SCHEMA_VERSION &&
            string(read(file, "status")) == "complete" &&
            Symbol(read(file, "protocol")) == study.protocol &&
            Float64(read(file, "rayleigh")) == study.rayleigh &&
            Symbol(read(file, "policy")) == :zero_action &&
            Int(read(file, "run_seed")) == options.run_seed &&
            Int(read(file, "steps")) == UNACTUATED_STEPS &&
            Int(read(file, "case_count")) == 8 &&
            string(read(file, "run_file_sha256")) == sources.run_file_sha256 &&
            string(read(file, "test_data_sha256")) == sources.test_data_sha256
        end
    catch
        false
    end
end

function configure_runtime!(options, study, sources, runtime_directory)
    ENV["REVISION_RUN_SEED"] = string(options.run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    include(sources.run_file_path)

    Float64(latest_binding(:RA)) == study.rayleigh || error(
        "Loaded run file has Ra=$(latest_binding(:RA)); expected $(study.rayleigh).",
    )
    abspath(string(latest_binding(:CORPUS_PATH))) == sources.test_data_path || error(
        "Loaded run file uses the wrong test corpus: $(latest_binding(:CORPUS_PATH)).",
    )
    runtime_hook = latest_binding(:hook)
    hasproperty(runtime_hook, :is_display_on_exit) &&
        (runtime_hook.is_display_on_exit = false)
    hasproperty(runtime_hook, :display_after_episode) &&
        (runtime_hook.display_after_episode = false)
    return nothing
end

function test_cases()
    corpus = latest_binding(:CORPUS)
    test_split = dict_value(corpus, :test)
    base_seeds = sort!(Int.(collect(keys(test_split))))
    length(base_seeds) == 2 || error(
        "Expected two Higher-Ra test basis snapshots, found $(length(base_seeds)).",
    )
    return vec([
        (
            case_id = "base_$(base_seed)_mirror_$(Int(mirror))_offset_$(offset)",
            choice = (; split = :test, base_seed, mirror, offset),
        )
        for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)
    ])
end

function reset_episode!(choice)
    initialize = latest_binding(:generate_random_init)
    Base.invokelatest(
        initialize;
        split = choice.split,
        base_seed = choice.base_seed,
        mirror = choice.mirror,
        offset = choice.offset,
    )
    runtime_rl = latest_binding(:RL)
    runtime_env = latest_binding(:env)
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    return nothing
end

function run_episode(case)
    reset_episode!(case.choice)
    runtime_env = latest_binding(:env)
    nusselt_function = latest_binding(:state_Nu)
    rewards = Vector{Float64}(undef, UNACTUATED_STEPS)
    state_nusselt = Vector{Float64}(undef, UNACTUATED_STEPS)
    actions = zeros(Float32, UNACTUATED_STEPS, 12)
    zero_action = zeros(Float32, 1, 12)

    for step in 1:UNACTUATED_STEPS
        Base.invokelatest(runtime_env, zero_action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        state_nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
    end
    all(isfinite, rewards) && all(isfinite, state_nusselt) || error(
        "Non-finite reward or state_Nu in $(case.case_id).",
    )
    return (;
        case_id = case.case_id,
        choice = case.choice,
        rewards,
        state_nusselt,
        actions,
        reward_sum = sum(rewards),
        mean_reward = mean(rewards),
        sum_state_nusselt = sum(state_nusselt),
        mean_state_nusselt = mean(state_nusselt),
    )
end

function atomic_save(path::AbstractString; values...)
    mkpath(dirname(path))
    temporary = joinpath(
        dirname(path),
        ".$(basename(path)).$(getpid()).$(uuid4()).tmp",
    )
    try
        JLD2.jldsave(temporary; values...)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return abspath(path)
end

function compute_one(options, study)
    sources = study_sources(study)
    output = baseline_path(options, study)
    if !options.overwrite && baseline_is_current(output, options, study, sources)
        println("Current unactuated baseline already exists: $output")
        return abspath(output)
    end

    mktempdir(; prefix = "higher-ra-$(study.tag)-unactuated-") do runtime_directory
        configure_runtime!(options, study, sources, runtime_directory)
        episodes = map(test_cases()) do case
            println("Running $(study.label) unactuated test: $(case.case_id)")
            episode = Base.invokelatest(run_episode, case)
            println(
                "  reward_sum=$(episode.reward_sum), " *
                "sum_state_Nu=$(episode.sum_state_nusselt)",
            )
            episode
        end
        reward_sums = getproperty.(episodes, :reward_sum)
        nusselt_sums = getproperty.(episodes, :sum_state_nusselt)
        atomic_save(
            output;
            schema_version = UNACTUATED_SCHEMA_VERSION,
            status = :complete,
            experiment = :higher_ra_unactuated_test_baseline,
            protocol = study.protocol,
            rayleigh = study.rayleigh,
            controller = :unactuated,
            policy = :zero_action,
            run_seed = options.run_seed,
            steps = UNACTUATED_STEPS,
            case_count = length(episodes),
            sources...,
            expert_path = "",
            expert_sha256 = "",
            source_checkpoint_path = "",
            source_checkpoint_sha256 = "",
            mean_reward_sum = mean(reward_sums),
            median_reward_sum = median(reward_sums),
            mean_sum_state_nusselt = mean(nusselt_sums),
            median_sum_state_nusselt = median(nusselt_sums),
            episodes,
            completed_at = string(Dates.now(Dates.UTC)),
        )
    end
    println("Saved unactuated baseline: $output")
    return abspath(output)
end

function run_study_subprocess(options, study)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_ROOT) $(@__FILE__) --run-one $(study.tag) --results-dir $(options.results_root) --run-seed $(options.run_seed)`
    options.overwrite && (command = `$command --overwrite`)
    run(command)
    return baseline_path(options, study)
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    if !isnothing(options.run_one)
        study = getproperty(UNACTUATED_STUDIES, options.run_one)
        return compute_one(options, study)
    end

    selected = options.selected_study === :all ? keys(UNACTUATED_STUDIES) :
        (options.selected_study,)
    outputs = String[]
    for tag in selected
        push!(outputs, run_study_subprocess(options, getproperty(UNACTUATED_STUDIES, tag)))
    end
    println("Completed $(length(outputs)) Higher-Ra unactuated baseline(s).")
    return outputs
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
