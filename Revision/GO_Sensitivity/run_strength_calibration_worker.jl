using Dates
using Printf

# This helper is intentionally launched in a fresh Julia process for each
# protocol/grouping combination. Expert_Apprentice.jl fixes both choices when
# it is included, so changing them inside one process would be misleading.

const P6_CALIBRATION_SEED = 600_601
const P6_CALIBRATION_UPDATES = 2_000
const P6_CALIBRATION_EVALUATION_START = 0
const P6_CALIBRATION_EVALUATION_INTERVAL = 25
const P6_CALIBRATION_RESUME_INTERVAL = 100
const P6_CALIBRATION_GARBAGE_COLLECTION_INTERVAL = 5
const P6_CALIBRATION_STRENGTHS = Dict(
    :fixed => [0.01, 0.03, 0.09],
    :varying => [0.003, 0.008, 0.025],
)

const P6_DIRECTORY = @__DIR__
const P6_PROJECT_ROOT = normpath(joinpath(P6_DIRECTORY, "..", ".."))
const P6_DISTILLATION_DIRECTORY = joinpath(
    P6_PROJECT_ROOT,
    "Revision",
    "Expert_Apprentice_Distillation",
)
const P6_CALIBRATION_RESULTS_ROOT = joinpath(
    P6_DIRECTORY,
    "results",
    "strength_calibration",
)

function calibration_worker_usage(io::IO = stdout)
    println(io, """
    Internal Package-6 calibration worker.

    Usage:
      julia --project=. run_strength_calibration_worker.jl \\
        --protocol fixed|varying --group-channels true|false

    Run run_strength_calibration_pilot.jl instead of calling this helper by hand.
    """)
end

function parse_calibration_worker_arguments(arguments)
    options = Dict{String, Any}(
        "protocol" => nothing,
        "group_channels" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            calibration_worker_usage()
            return nothing
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
    protocol = Symbol(lowercase(string(options["protocol"])))
    protocol in (:fixed, :varying) || error("--protocol must be fixed or varying.")

    isnothing(options["group_channels"]) && error("--group-channels is required.")
    group_channels_text = lowercase(string(options["group_channels"]))
    group_channels_text in ("true", "false") || error(
        "--group-channels must be true or false.",
    )
    return (
        protocol,
        group_channels = group_channels_text == "true",
    )
end

function calibration_strength_tag(value::Real)
    return replace(@sprintf("%.8g", Float64(value)), "." => "p", "-" => "m")
end

function calibration_grouping_tag(group_channels_value::Bool)
    return group_channels_value ? "grouped_channels" : "separate_channels"
end

function configure_calibration_environment!(protocol::Symbol, group_channels_value::Bool)
    grouping_tag = calibration_grouping_tag(group_channels_value)
    expert_path = joinpath(
        P6_DISTILLATION_DIRECTORY,
        "experts",
        string(protocol),
        "agent.jld2",
    )
    worker_directory = joinpath(P6_DISTILLATION_DIRECTORY, "worker_results")
    runtime_directory = joinpath(
        P6_DIRECTORY,
        "runtime",
        "strength_calibration_$(protocol)_$(grouping_tag)",
    )

    ENV["DISTILLATION_PROTOCOL"] = string(protocol)
    ENV["DISTILLATION_AUTOLOAD_PROTOCOL"] = string(protocol)
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "false"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(group_channels_value)
    ENV["REVISION_RUN_SEED"] = string(P6_CALIBRATION_SEED)
    ENV[protocol === :fixed ? "DISTILLATION_FIXED_EXPERT_PATH" : "DISTILLATION_VARYING_EXPERT_PATH"] = expert_path
    ENV["DISTILLATION_WORKER_DIRECTORY"] = worker_directory
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(
        P6_CALIBRATION_RESULTS_ROOT,
        "apprentice_outputs",
        string(protocol),
        grouping_tag,
    )
    return (expert_path, worker_directory, runtime_directory)
end

function assert_calibration_inputs(protocol::Symbol, expert_path::AbstractString)
    isfile(expert_path) || error("Expert checkpoint is missing: $expert_path")
    coverages = Dict(
        split => distillation_coverage(protocol, split)
        for split in (:train, :validation, :test)
    )
    for split in (:train, :validation, :test)
        coverage = coverages[split]
        coverage.complete || error(
            "Incomplete $protocol/$split distillation corpus: expected " *
            "$(coverage.expected), found $(coverage.actual).",
        )
    end

    datasets = Dict(
        split => distillation_dataset(protocol, split)
        for split in (:train, :validation, :test)
    )
    identifiers = Set(string(datasets[split][:expert_identifier]) for split in keys(datasets))
    length(identifiers) == 1 || error(
        "$protocol corpus splits were generated by different experts: $identifiers",
    )
    corpus_identifier = only(identifiers)
    loaded_identifier = string(DISTILLATION_EXPERT_METADATA[:identifier])
    corpus_identifier == loaded_identifier || error(
        "$protocol corpus was generated by $corpus_identifier, but the selected expert is " *
        "$loaded_identifier. Regenerate all three splits with the selected checkpoint.",
    )
    if protocol === :fixed
        (datasets[:train] === datasets[:validation] && datasets[:validation] === datasets[:test]) ||
            error("Fixed IC must expose one shared train/validation/test dataset object.")
    else
        length(Set(objectid(datasets[split]) for split in keys(datasets))) == 3 || error(
            "Varying IC requires distinct train, validation, and test dataset objects.",
        )
    end
    return (coverages, datasets, corpus_identifier)
end

function calibration_run_id(
    protocol::Symbol,
    group_channels_value::Bool,
    strength::Real,
)
    grouping_tag = calibration_grouping_tag(group_channels_value)
    return "calibration_$(protocol)_$(grouping_tag)_strength_" *
           "$(calibration_strength_tag(strength))_seed_$(P6_CALIBRATION_SEED)_" *
           "updates_$(P6_CALIBRATION_UPDATES)_interval_$(P6_CALIBRATION_EVALUATION_INTERVAL)"
end

function calibration_archive_config(
    protocol::Symbol,
    group_channels_value::Bool,
    strength::Real,
    training_config,
    expert_path::AbstractString,
    corpus_identifier::AbstractString,
    datasets,
)
    return Dict{Symbol, Any}(
        :experiment => :package6_strength_calibration_pilot,
        :scientific_result => false,
        :purpose => :choose_five_point_production_strength_grid,
        :protocol => protocol,
        :method => :go,
        :group_rows_by_overlap => true,
        :group_channels => group_channels_value,
        :grouping => Symbol(calibration_grouping_tag(group_channels_value)),
        :apprentice_seed => P6_CALIBRATION_SEED,
        :batch_order_seed => P6_CALIBRATION_SEED + 10_000,
        :calibration_strengths => copy(P6_CALIBRATION_STRENGTHS[protocol]),
        :regularization_strength => Float64(strength),
        :regularized_updates => training_config.regularized_updates,
        :post_pruning_finetune_updates => training_config.post_pruning_finetune_updates,
        :batch_size => training_config.batch_size,
        :proximal_interval => training_config.proximal_interval,
        :validation_batch_size => training_config.validation_batch_size,
        :validation_prediction_mode => training_config.validation_prediction_mode,
        :diagnostic_teacher_forced => training_config.diagnostic_teacher_forced,
        :hard_threshold_candidates => false,
        :expert_identifier => corpus_identifier,
        :expert_path => abspath(expert_path),
        :train_source_files => abspath.(String.(datasets[:train][:source_files])),
        :validation_source_files => abspath.(String.(datasets[:validation][:source_files])),
        :test_source_files => abspath.(String.(datasets[:test][:source_files])),
        :train_sample_count => Int(datasets[:train][:sample_count]),
        :validation_sample_count => Int(datasets[:validation][:sample_count]),
        :test_sample_count => Int(datasets[:test][:sample_count]),
        :test_data_used => false,
    )
end

function save_calibration_summary!(manager, result, elapsed_seconds::Real)
    path = joinpath(manager.run_directory, "calibration_summary.jld2")
    pareto_atomic_save(
        path;
        experiment = :package6_strength_calibration_pilot,
        scientific_result = false,
        completed_at = string(Dates.now()),
        elapsed_seconds = Float64(elapsed_seconds),
        update_count = length(result.losses),
        final_training_loss = isempty(result.losses) ? NaN : last(result.losses),
        regularization_strength = result.regularization_strength,
        evaluation_count = manager.evaluation_count,
        pareto_front = manager.front,
    )
    return path
end

function run_calibration_strength!(
    protocol::Symbol,
    group_channels_value::Bool,
    strength::Real,
    initial_apprentice,
    expert_path::AbstractString,
    corpus_identifier::AbstractString,
    coverages,
    datasets,
)
    run_id = calibration_run_id(protocol, group_channels_value, strength)
    run_directory = joinpath(
        P6_CALIBRATION_RESULTS_ROOT,
        string(protocol),
        calibration_grouping_tag(group_channels_value),
        run_id,
    )
    training_config = ApprenticeTrainingConfig(
        regularized_updates = P6_CALIBRATION_UPDATES,
        post_pruning_finetune_updates = 0,
        batch_size = protocol === :fixed ? 20 : 100,
        proximal_interval = 1,
        reweight_interval = 10,
        regularization_strength = Float64(strength),
        validation_batch_size = protocol === :fixed ? 200 : 512,
        validation_prediction_mode = :autoregressive,
        diagnostic_teacher_forced = false,
    )
    schedule = CandidateSchedule(
        start_update = P6_CALIBRATION_EVALUATION_START,
        evaluation_interval = P6_CALIBRATION_EVALUATION_INTERVAL,
        garbage_collection_interval = P6_CALIBRATION_GARBAGE_COLLECTION_INTERVAL,
        resume_interval = P6_CALIBRATION_RESUME_INTERVAL,
    )
    manager = initialize_pareto_archive(
        run_directory;
        run_id,
        schedule,
        config = calibration_archive_config(
            protocol,
            group_channels_value,
            strength,
            training_config,
            expert_path,
            corpus_identifier,
            datasets,
        ),
    )
    resume_checkpoint = load_resume_checkpoint(manager)
    if !isnothing(resume_checkpoint) && resume_checkpoint.status === :complete
        println("Skipping completed calibration run: $run_id")
        return manager
    elseif !isnothing(resume_checkpoint) && resume_checkpoint.status === :failed
        message = get(resume_checkpoint.resume_state, :failure_message, "unknown numerical failure")
        error("Calibration run $run_id previously failed at update $(resume_checkpoint.update): $message")
    end
    resume = !isnothing(resume_checkpoint)

    println()
    println("Package-6 one-seed strength calibration")
    println("  protocol:                $protocol")
    println("  grouping:                $(calibration_grouping_tag(group_channels_value))")
    println("  strength:                $strength")
    println("  train coverage:          $(coverages[:train].actual)")
    println("  validation coverage:     $(coverages[:validation].actual)")
    println("  test coverage (unused):  $(coverages[:test].actual)")
    println("  resume:                  $resume")
    println("  output:                  $run_directory")

    started_at = time()
    result = train_apprentice!(
        deepcopy(initial_apprentice);
        method = :go,
        train_dataset = datasets[:train],
        validation_dataset = datasets[:validation],
        config = training_config,
        archive_manager = manager,
        threshold_specs = HardThresholdSpec[],
        group_rows_by_overlap = true,
        group_channels = group_channels_value,
        training_rng = StableRNG(P6_CALIBRATION_SEED + 10_000),
        resume,
    )
    elapsed_seconds = time() - started_at
    summary_path = save_calibration_summary!(manager, result, elapsed_seconds)
    @printf("Completed %s in %.2f seconds.\n", run_id, elapsed_seconds)
    println("Summary: $summary_path")
    return manager
end

function calibration_worker_main(arguments = ARGS)
    options = parse_calibration_worker_arguments(arguments)
    isnothing(options) && return nothing
    expert_path, _, _ = configure_calibration_environment!(
        options.protocol,
        options.group_channels,
    )
    include(joinpath(P6_DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
    return Base.invokelatest(
        run_loaded_calibration_worker,
        options,
        expert_path,
    )
end

function run_loaded_calibration_worker(options, expert_path::AbstractString)
    coverages, datasets, corpus_identifier = assert_calibration_inputs(
        options.protocol,
        expert_path,
    )
    initial_apprentice = deepcopy(apprentice)

    for strength in P6_CALIBRATION_STRENGTHS[options.protocol]
        run_calibration_strength!(
            options.protocol,
            options.group_channels,
            strength,
            initial_apprentice,
            expert_path,
            corpus_identifier,
            coverages,
            datasets,
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        calibration_worker_main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
