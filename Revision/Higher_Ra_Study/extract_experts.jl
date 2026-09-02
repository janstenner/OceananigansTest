ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Dates
using JLD2
using PlotlyJS
using SHA
using Statistics
using UUIDs

include(joinpath(@__DIR__, "..", "MAT_expert_training", "MATExpertTraining.jl"))
using .MATExpertTraining

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ROLLING_WINDOW = 100
const TEST_STEPS = 200
const SCHEMA_VERSION = 1
const CURVE_COLOR = "#277DA1"
const RUN_COLOR = "rgba(39, 125, 161, 0.22)"
const RIBBON_COLOR = "rgba(39, 125, 161, 0.18)"

const STUDIES = (
    ra5e4 = (
        tag = :ra5e4,
        label = "Ra=5e4",
        protocol = :varying_ra5e4,
        rayleigh = 5.0e4,
        default_results = MATExpertTraining.DEFAULT_RA5E4_RESULTS_DIRECTORY,
        run_file = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "VaryingIC_MAT_Ra5e4.jl"),
        corpus_file = joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus_Ra5e4.jld2"),
    ),
    ra1e5 = (
        tag = :ra1e5,
        label = "Ra=1e5",
        protocol = :varying_ra1e5,
        rayleigh = 1.0e5,
        default_results = MATExpertTraining.DEFAULT_RA1E5_RESULTS_DIRECTORY,
        run_file = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "VaryingIC_MAT_Ra1e5.jl"),
        corpus_file = joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus_Ra1e5.jld2"),
    ),
)

function usage(io::IO = stdout)
    println(io, """
    Usage:
      julia --startup-file=no --project=. Revision/Higher_Ra_Study/extract_experts.jl [options]

    Options:
      --ra5e4-results PATH      Ra=5e4 training-result root.
      --ra1e5-results PATH      Ra=1e5 training-result root.
      --output-dir PATH         Output root (default: Revision/Higher_Ra_Study).
      --skip-validation         Extract experts and plots without test baselines.
      --overwrite-baselines     Recompute matching existing test baselines.
      --help

    For each Rayleigh number, every readable best-so-far, final, and current
    resume checkpoint below the result root is considered. The checkpoint with
    the largest mean reward over its latest 100 completed episodes is selected.
    """)
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "ra5e4_results" => STUDIES.ra5e4.default_results,
        "ra1e5_results" => STUDIES.ra1e5.default_results,
        "output_dir" => @__DIR__,
        "skip_validation" => false,
        "overwrite_baselines" => false,
        "validate_one" => nothing,
        "expert_path" => nothing,
        "baseline_path" => nothing,
        "checkpoint_path" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument in ("--skip-validation", "--overwrite-baselines")
            options[replace(argument[3:end], "-" => "_")] = true
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
    return (;
        ra5e4_results = abspath(string(options["ra5e4_results"])),
        ra1e5_results = abspath(string(options["ra1e5_results"])),
        output_dir = abspath(string(options["output_dir"])),
        skip_validation = Bool(options["skip_validation"]),
        overwrite_baselines = Bool(options["overwrite_baselines"]),
        validate_one = isnothing(options["validate_one"]) ? nothing :
            Symbol(lowercase(string(options["validate_one"]))),
        expert_path = isnothing(options["expert_path"]) ? nothing :
            abspath(string(options["expert_path"])),
        baseline_path = isnothing(options["baseline_path"]) ? nothing :
            abspath(string(options["baseline_path"])),
        checkpoint_path = isnothing(options["checkpoint_path"]) ? nothing :
            abspath(string(options["checkpoint_path"])),
    )
end

file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
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

optional_read(file, key, default) = haskey(file, key) ? read(file, key) : default

function checkpoint_kind(path)
    name = basename(path)
    name == "best_so_far.jld2" && return :best_so_far
    name == "final.jld2" && return :final
    name == "latest.jld2" && basename(dirname(path)) == "resume" && return :resume
    return nothing
end

function checkpoint_paths(results_root)
    isdir(results_root) || error("Training-result root does not exist: $results_root")
    paths = String[]
    for (directory, _, files) in walkdir(results_root)
        for filename in files
            path = joinpath(directory, filename)
            isnothing(checkpoint_kind(path)) || push!(paths, abspath(path))
        end
    end
    return sort!(unique!(paths))
end

function read_checkpoint(path, study)
    kind = checkpoint_kind(path)
    isnothing(kind) && return nothing
    return JLD2.jldopen(path, "r") do file
        haskey(file, "agent") || return nothing
        stored_protocol = Symbol(optional_read(file, "protocol", :unknown))
        stored_rayleigh = Float64(optional_read(file, "rayleigh", NaN))
        stored_protocol == study.protocol || return nothing
        stored_rayleigh == study.rayleigh || return nothing
        rewards = Float64.(optional_read(file, "rewards", Float64[]))
        length(rewards) >= ROLLING_WINDOW || return nothing
        criterion = mean(@view rewards[(end - ROLLING_WINDOW + 1):end])
        isfinite(criterion) || return nothing
        recorded_criterion = Float64(optional_read(file, "criterion_value", criterion))
        kind === :best_so_far && isapprox(
            criterion,
            recorded_criterion;
            atol = 1e-8,
            rtol = 1e-10,
        ) || kind !== :best_so_far || error(
            "Recorded rolling-100 criterion does not match rewards in $path.",
        )
        return (;
            path = abspath(path),
            kind,
            status = string(optional_read(file, "status", "unknown")),
            protocol = stored_protocol,
            rayleigh = stored_rayleigh,
            run_id = string(optional_read(file, "run_id", basename(dirname(path)))),
            run_index = Int(optional_read(file, "run_index", 0)),
            run_seed = Int(optional_read(file, "run_seed", -1)),
            ic_seed = Int(optional_read(file, "ic_seed", -1)),
            config_fingerprint = string(optional_read(file, "config_fingerprint", "")),
            episodes_completed = Int(optional_read(file, "episodes_completed", length(rewards))),
            criterion,
            rewards,
        )
    end
end

function available_checkpoints(results_root, study)
    records = NamedTuple[]
    for path in checkpoint_paths(results_root)
        record = try
            read_checkpoint(path, study)
        catch error_value
            @warn "Skipping unreadable or inconsistent checkpoint." path exception=(error_value, catch_backtrace())
            nothing
        end
        isnothing(record) || push!(records, record)
    end
    isempty(records) && error(
        "No checkpoint with at least $ROLLING_WINDOW rewards was found for $(study.label) below $results_root.",
    )
    sort!(records; by = record -> (record.criterion, record.episodes_completed, record.path), rev = true)
    return records
end

function curve_records(checkpoints)
    by_run_directory = Dict{String, NamedTuple}()
    for record in checkpoints
        record.kind === :best_so_far && continue
        run_directory = record.kind === :resume ? dirname(dirname(record.path)) : dirname(record.path)
        key = abspath(run_directory)
        if !haskey(by_run_directory, key) ||
           (record.kind === :final && by_run_directory[key].kind !== :final)
            by_run_directory[key] = record
        end
    end
    records = collect(values(by_run_directory))
    sort!(records; by = record -> (record.run_index, record.run_id, record.path))
    isempty(records) && error("No per-run final or resume curves are available.")
    return records
end

rolling_mean(values, width = ROLLING_WINDOW) = length(values) < width ? Float64[] : [
    mean(@view values[(index - width + 1):index]) for index in width:length(values)
]

function curve_statistics(records)
    curves = [rolling_mean(record.rewards) for record in records]
    filter!(curve -> !isempty(curve), curves)
    isempty(curves) && error("No run has a complete rolling-$ROLLING_WINDOW window.")
    count = minimum(length, curves)
    matrix = reduce(hcat, (curve[1:count] for curve in curves))
    return [
        begin
            values = vec(matrix[index, :])
            (
                episode = index + ROLLING_WINDOW - 1,
                n = length(values),
                mean = mean(values),
                median = median(values),
                q25 = quantile(values, 0.25),
                q75 = quantile(values, 0.75),
            )
        end for index in 1:count
    ]
end

function plot_layout(title)
    return Layout(
        template = "plotly_white",
        title = attr(
            text = title,
            x = 0.5,
            xanchor = "center",
            font = attr(size = 22, color = "#252525"),
        ),
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        width = 900,
        height = 560,
        margin = attr(l = 100, r = 30, t = 80, b = 85),
        font = attr(family = "Arial, sans-serif", size = 15, color = "#303030"),
        xaxis = attr(
            title = attr(text = "Episode", standoff = 12),
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            linewidth = 1,
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        yaxis = attr(
            title = attr(text = "Score (rolling mean, window=$ROLLING_WINDOW)", standoff = 12),
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            linewidth = 1,
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        showlegend = true,
        legend = attr(
            x = 0.985,
            y = 0.02,
            xanchor = "right",
            yanchor = "bottom",
            bgcolor = "rgba(255, 255, 255, 0.92)",
            bordercolor = "#CFCFCF",
            borderwidth = 1,
            font = attr(size = 13),
        ),
    )
end

function save_learning_curves(study, records, output_directory)
    mkpath(output_directory)
    statistics = curve_statistics(records)
    traces = PlotlyJS.GenericTrace[]
    for (index, record) in enumerate(records)
        curve = rolling_mean(record.rewards)
        push!(traces, scatter(
            x = collect(ROLLING_WINDOW:(ROLLING_WINDOW + length(curve) - 1)),
            y = curve,
            mode = "lines",
            name = "MAT runs (n=$(length(records)))",
            legendgroup = "runs",
            showlegend = index == 1,
            line = attr(color = RUN_COLOR, width = 1),
            hovertemplate = "$(record.run_id)<br>Episode %{x}<br>Score %{y:.2f}<extra></extra>",
        ))
    end
    episodes = getproperty.(statistics, :episode)
    push!(traces, scatter(
        x = episodes,
        y = getproperty.(statistics, :q25),
        mode = "lines",
        line = attr(width = 0),
        hoverinfo = "skip",
        showlegend = false,
    ))
    push!(traces, scatter(
        x = episodes,
        y = getproperty.(statistics, :q75),
        mode = "lines",
        line = attr(width = 0),
        fill = "tonexty",
        fillcolor = RIBBON_COLOR,
        hoverinfo = "skip",
        showlegend = false,
    ))
    push!(traces, scatter(
        x = episodes,
        y = getproperty.(statistics, :median),
        mode = "lines",
        name = "MAT median + IQR",
        line = attr(color = CURVE_COLOR, width = 3),
        hovertemplate = "Episode %{x}<br>Median %{y:.2f}<extra>%{fullData.name}</extra>",
    ))
    push!(traces, scatter(
        x = episodes,
        y = getproperty.(statistics, :mean),
        mode = "lines",
        name = "MAT mean",
        line = attr(color = CURVE_COLOR, width = 2, dash = "dash"),
        hovertemplate = "Episode %{x}<br>Mean %{y:.2f}<extra>%{fullData.name}</extra>",
    ))
    aggregate_path = joinpath(output_directory, "$(study.tag)_learning_curves.svg")
    PlotlyJS.savefig(
        Plot(traces, plot_layout("$(study.label) MAT learning curves")),
        aggregate_path;
        width = 900,
        height = 560,
    )

    individual_traces = PlotlyJS.GenericTrace[]
    for record in records
        curve = rolling_mean(record.rewards)
        push!(individual_traces, scatter(
            x = collect(ROLLING_WINDOW:(ROLLING_WINDOW + length(curve) - 1)),
            y = curve,
            mode = "lines",
            name = record.run_id,
            line = attr(color = CURVE_COLOR, width = 1.5),
            opacity = 0.55,
            hovertemplate = "$(record.run_id)<br>Episode %{x}<br>Score %{y:.2f}<extra></extra>",
        ))
    end
    individual_path = joinpath(output_directory, "$(study.tag)_individual_curves.svg")
    PlotlyJS.savefig(
        Plot(individual_traces, plot_layout("$(study.label) MAT individual runs")),
        individual_path;
        width = 900,
        height = 560,
    )
    return (aggregate = abspath(aggregate_path), individual = abspath(individual_path))
end

function publish_expert(study, winner, output_directory)
    target = joinpath(output_directory, "experts", string(study.tag), "expert.jld2")
    MATExpertTraining.save_compact_expert_from_checkpoint!(winner.path, target)
    return (;
        rayleigh = study.rayleigh,
        protocol = study.protocol,
        run_id = winner.run_id,
        run_index = winner.run_index,
        run_seed = winner.run_seed,
        ic_seed = winner.ic_seed,
        rolling_window = ROLLING_WINDOW,
        criterion_value = winner.criterion,
        episodes_completed = winner.episodes_completed,
        checkpoint_kind = winner.kind,
        source_checkpoint_path = winner.path,
        source_checkpoint_sha256 = file_sha256(winner.path),
        expert_path = abspath(target),
        expert_sha256 = file_sha256(target),
    )
end

function dict_value(mapping, key::Symbol)
    haskey(mapping, key) && return mapping[key]
    haskey(mapping, string(key)) && return mapping[string(key)]
    error("Missing '$key'.")
end

latest_binding(name::Symbol) = Base.invokelatest(() -> getfield(@__MODULE__, name))

function validation_cases()
    corpus = latest_binding(:CORPUS)
    test_split = dict_value(corpus, :test)
    base_seeds = sort!(Int.(collect(keys(test_split))))
    length(base_seeds) == 2 || error("Expected two test basis snapshots, found $(length(base_seeds)).")
    return vec([
        (
            case_id = "base_$(base_seed)_mirror_$(Int(mirror))_offset_$(offset)",
            choice = (split = :test, base_seed, mirror, offset),
        )
        for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)
    ])
end

function reset_validation_episode!(choice)
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

function deterministic_action()
    runtime_rl = latest_binding(:RL)
    runtime_agent = latest_binding(:agent)
    runtime_env = latest_binding(:env)
    action = Base.invokelatest(runtime_rl.prob, runtime_agent.policy, runtime_env).μ
    hasproperty(runtime_agent.policy, :clip1) && runtime_agent.policy.clip1 &&
        clamp!(action, -1.0f0, 1.0f0)
    return action
end

function run_validation_episode(case)
    reset_validation_episode!(case.choice)
    runtime_env = latest_binding(:env)
    nusselt_function = latest_binding(:state_Nu)
    rewards = Vector{Float64}(undef, TEST_STEPS)
    state_nusselt = Vector{Float64}(undef, TEST_STEPS)
    actions = Matrix{Float32}(undef, TEST_STEPS, 12)
    for step in 1:TEST_STEPS
        action = Base.invokelatest(deterministic_action)
        action_values = vec(Float32.(Array(action)))
        length(action_values) == 12 || error("Expected 12 actions, got $(length(action_values)).")
        actions[step, :] .= action_values
        Base.invokelatest(runtime_env, action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        state_nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
    end
    all(isfinite, rewards) && all(isfinite, state_nusselt) || error(
        "Non-finite result in $(case.case_id).",
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

function baseline_is_current(path, study, expert_sha, checkpoint_sha)
    isfile(path) || return false
    return try
        JLD2.jldopen(path, "r") do file
            Int(read(file, "schema_version")) == SCHEMA_VERSION &&
            string(read(file, "status")) == "complete" &&
            Symbol(read(file, "protocol")) == study.protocol &&
            Float64(read(file, "rayleigh")) == study.rayleigh &&
            string(read(file, "expert_sha256")) == expert_sha &&
            string(read(file, "source_checkpoint_sha256")) == checkpoint_sha
        end
    catch
        false
    end
end

function validate_one(options)
    options.validate_one in keys(STUDIES) || error(
        "--validate-one must be ra5e4 or ra1e5.",
    )
    isnothing(options.expert_path) && error("--expert-path is required with --validate-one.")
    isnothing(options.baseline_path) && error("--baseline-path is required with --validate-one.")
    isnothing(options.checkpoint_path) && error("--checkpoint-path is required with --validate-one.")
    study = getproperty(STUDIES, options.validate_one)
    isfile(options.expert_path) || error("Expert does not exist: $(options.expert_path)")
    isfile(options.checkpoint_path) || error("Source checkpoint does not exist: $(options.checkpoint_path)")
    isfile(study.run_file) || error("Run file does not exist: $(study.run_file)")
    isfile(study.corpus_file) || error("Test corpus does not exist: $(study.corpus_file)")
    expert_sha = file_sha256(options.expert_path)
    checkpoint_sha = file_sha256(options.checkpoint_path)
    if !options.overwrite_baselines && baseline_is_current(
        options.baseline_path,
        study,
        expert_sha,
        checkpoint_sha,
    )
        println("Current test baseline already exists: $(options.baseline_path)")
        return options.baseline_path
    end
    mktempdir(; prefix = "higher-ra-$(study.tag)-test-") do runtime_directory
        ENV["REVISION_RUN_SEED"] = "20260902"
        ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
        ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
        include(study.run_file)
        global agent = JLD2.load(options.expert_path, "agent")
        hasproperty(hook, :is_display_on_exit) && (hook.is_display_on_exit = false)
        hasproperty(hook, :display_after_episode) && (hook.display_after_episode = false)
        cases = validation_cases()
        episodes = map(cases) do case
            println("Running $(study.label) expert test: $(case.case_id)")
            run_validation_episode(case)
        end
        reward_sums = getproperty.(episodes, :reward_sum)
        nusselt_sums = getproperty.(episodes, :sum_state_nusselt)
        atomic_save(
            options.baseline_path;
            schema_version = SCHEMA_VERSION,
            status = :complete,
            experiment = :higher_ra_expert_test_baseline,
            protocol = study.protocol,
            rayleigh = study.rayleigh,
            policy = :deterministic_mean_action,
            steps = TEST_STEPS,
            case_count = length(episodes),
            run_file_path = abspath(study.run_file),
            run_file_sha256 = file_sha256(study.run_file),
            test_data_path = abspath(study.corpus_file),
            test_data_sha256 = file_sha256(study.corpus_file),
            expert_path = options.expert_path,
            expert_sha256 = expert_sha,
            source_checkpoint_path = options.checkpoint_path,
            source_checkpoint_sha256 = checkpoint_sha,
            mean_reward_sum = mean(reward_sums),
            median_reward_sum = median(reward_sums),
            mean_sum_state_nusselt = mean(nusselt_sums),
            median_sum_state_nusselt = median(nusselt_sums),
            episodes,
            completed_at = string(Dates.now(Dates.UTC)),
        )
    end
    println("Saved test baseline: $(options.baseline_path)")
    return options.baseline_path
end

function run_validation_subprocess(study, publication, baseline_path, overwrite)
    command = `$(Base.julia_cmd()) --startup-file=no --project=$(PROJECT_ROOT) $(@__FILE__) --validate-one $(study.tag) --expert-path $(publication.expert_path) --checkpoint-path $(publication.source_checkpoint_path) --baseline-path $baseline_path`
    overwrite && (command = `$command --overwrite-baselines`)
    run(command)
    return abspath(baseline_path)
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    !isnothing(options.validate_one) && return validate_one(options)

    mkpath(options.output_dir)
    plot_directory = joinpath(options.output_dir, "plots")
    publications = NamedTuple[]
    plot_records = NamedTuple[]
    for study in values(STUDIES)
        results_root = study.tag === :ra5e4 ? options.ra5e4_results : options.ra1e5_results
        checkpoints = available_checkpoints(results_root, study)
        winner = first(checkpoints)
        println(
            "Selected $(study.label): $(winner.run_id), " *
            "rolling-100 mean=$(winner.criterion), checkpoint=$(winner.path)",
        )
        publication = publish_expert(study, winner, options.output_dir)
        push!(publications, publication)
        curves = curve_records(checkpoints)
        plots = save_learning_curves(study, curves, plot_directory)
        push!(plot_records, (;
            protocol = study.protocol,
            rayleigh = study.rayleigh,
            run_count = length(curves),
            aggregate_svg = plots.aggregate,
            individual_svg = plots.individual,
        ))
    end

    baseline_records = NamedTuple[]
    if !options.skip_validation
        for publication in publications
            study = publication.protocol === STUDIES.ra5e4.protocol ? STUDIES.ra5e4 : STUDIES.ra1e5
            baseline_path = joinpath(
                options.output_dir,
                "Baselines",
                string(study.tag),
                "expert.jld2",
            )
            path = run_validation_subprocess(
                study,
                publication,
                baseline_path,
                options.overwrite_baselines,
            )
            push!(baseline_records, (;
                protocol = study.protocol,
                rayleigh = study.rayleigh,
                path,
                sha256 = file_sha256(path),
            ))
        end
    end

    manifest_path = joinpath(options.output_dir, "experts", "selection_manifest.jld2")
    atomic_save(
        manifest_path;
        schema_version = SCHEMA_VERSION,
        status = :complete,
        experiment = :higher_ra_expert_extraction,
        selection_basis = :maximum_latest_100_episode_mean_reward,
        rolling_window = ROLLING_WINDOW,
        publications,
        plots = plot_records,
        baselines = baseline_records,
        source_roots = (
            ra5e4 = options.ra5e4_results,
            ra1e5 = options.ra1e5_results,
        ),
        completed_at = string(Dates.now(Dates.UTC)),
    )
    println("Saved selection manifest: $manifest_path")
    return manifest_path
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
