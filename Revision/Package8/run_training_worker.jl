using Dates
using JLD2
using Printf
using SHA
using StableRNGs

include(joinpath(@__DIR__, "Package8Study.jl"))
using .Package8Study

const P8_DIRECTORY = @__DIR__
const P8_PROJECT_ROOT = normpath(joinpath(P8_DIRECTORY, "..", ".."))
const P8_DISTILLATION_DIRECTORY = joinpath(P8_PROJECT_ROOT, "Revision", "Expert_Apprentice_Distillation")
const P8_DEFAULT_RESULTS_ROOT = joinpath(P8_DIRECTORY, "results")

function worker_usage(io::IO = stdout)
    println(io, """
    Package-8 Varying-IC training worker.

    Usage:
      julia --project=. run_training_worker.jl --experiment-id ID --config NAME --strength VALUE \\
        --replicate 1|2|3 [--results-dir PATH] [--retry-failed]
        [--smoke-updates N]
    """)
end

function parse_worker_arguments(arguments)
    values = Dict{String, Any}(
        "experiment_id" => nothing,
        "config" => nothing,
        "strength" => nothing,
        "replicate" => nothing,
        "results_dir" => P8_DEFAULT_RESULTS_ROOT,
        "retry_failed" => false,
        "smoke_updates" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            worker_usage()
            return nothing
        elseif argument == "--retry-failed"
            values["retry_failed"] = true
            index += 1
        elseif startswith(argument, "--")
            index == length(arguments) && error("Missing value after $argument.")
            key = replace(argument[3:end], "-" => "_")
            haskey(values, key) || error("Unknown option '$argument'.")
            values[key] = arguments[index + 1]
            index += 2
        else
            error("Unknown argument '$argument'.")
        end
    end
    for key in ("experiment_id", "config", "strength", "replicate")
        isnothing(values[key]) && error("--$(replace(key, "_" => "-")) is required.")
    end
    strength = parse(Float64, string(values["strength"]))
    replicate = parse(Int, string(values["replicate"]))
    smoke_updates = isnothing(values["smoke_updates"]) ? nothing : parse(Int, string(values["smoke_updates"]))
    !isnothing(smoke_updates) && smoke_updates < 0 && error("--smoke-updates must be nonnegative.")
    updates = isnothing(smoke_updates) ? P8_UPDATES : smoke_updates
    return (
        job = job_for(values["experiment_id"], values["config"], strength, replicate; updates),
        results_root = abspath(string(values["results_dir"])),
        retry_failed = Bool(values["retry_failed"]),
        smoke_updates,
    )
end

function configure_worker_environment!(options)
    job = options.job
    directory = run_directory(options.results_root, job)
    expert_path = joinpath(P8_DISTILLATION_DIRECTORY, "experts", "varying", "agent.jld2")
    ENV["DISTILLATION_PROTOCOL"] = "varying"
    ENV["DISTILLATION_AUTOLOAD_PROTOCOL"] = "varying"
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "false"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(job.group_channels)
    ENV["REVISION_RUN_SEED"] = string(job.apprentice_seed)
    ENV["DISTILLATION_VARYING_EXPERT_PATH"] = expert_path
    # Restrict discovery to the canonical Varying corpus directory so archived
    # worker_results/old files cannot introduce duplicate worker identities.
    ENV["DISTILLATION_WORKER_DIRECTORY"] = joinpath(P8_DISTILLATION_DIRECTORY, "worker_results", "varying")
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(directory, "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(directory, "apprentice_output")
    return (directory, expert_path)
end

function assert_varying_inputs(expert_path::AbstractString)
    isfile(expert_path) || error("Varying expert checkpoint is missing: $expert_path")
    train_coverage = distillation_coverage(:varying, :train)
    train_coverage.complete || error(
        "Incomplete varying/train distillation corpus: expected $(train_coverage.expected), found $(train_coverage.actual).",
    )
    validation_coverage = distillation_coverage(:varying, :validation)
    validation_coverage.complete || error(
        "Incomplete varying/validation distillation corpus: expected $(validation_coverage.expected), found $(validation_coverage.actual).",
    )
    train_dataset = distillation_dataset(:varying, :train)
    validation_dataset = distillation_dataset(:varying, :validation)
    train_dataset === validation_dataset && error("Varying training and validation must use distinct corpus objects.")
    corpus_identifier = string(train_dataset[:expert_identifier])
    validation_identifier = string(validation_dataset[:expert_identifier])
    validation_identifier == corpus_identifier || error(
        "Varying train and validation corpora use different experts.",
    )
    loaded_identifier = string(DISTILLATION_EXPERT_METADATA[:identifier])
    corpus_identifier == loaded_identifier || error(
        "Varying corpus expert $corpus_identifier does not match loaded expert $loaded_identifier.",
    )
    return (; train_coverage, validation_coverage, train_dataset, validation_dataset, corpus_identifier)
end

function parameter_hash(model)
    io = IOBuffer()
    parameters = Flux.trainables(model)
    write(io, string(length(parameters)), UInt8(0))
    for parameter in parameters
        array = Array(parameter)
        write(io, string(eltype(array)), UInt8(0), string(size(array)), UInt8(0))
        write(io, reinterpret(UInt8, vec(array)))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function threshold_specs()
    return [
        HardThresholdSpec(Symbol("threshold_", replace(string(value), "." => "p")), :absolute, value;
                          analysis_scope = :package8)
        for value in P8_THRESHOLDS[2:end]
    ]
end

function archive_config(options, training_config, expert_path, inputs, initial_hash)
    job = options.job
    return Dict{Symbol, Any}(
        :schema_version => P8_SCHEMA_VERSION,
        :experiment => :package8_varying_regularizer_comparison,
        :experiment_id => job.experiment_id,
        :scientific_scope => isnothing(options.smoke_updates) ? :package8 : :smoke_test,
        :protocol => :varying,
        :training_data_split => :train,
        :validation_data_split => :validation,
        :test_data_used_during_training => false,
        :configuration => job.configuration,
        :method => job.method,
        :grouping => job.grouping,
        :group_rows_by_overlap => true,
        :group_channels => job.group_channels,
        :regularization_strength => job.regularization_strength,
        :replicate => job.replicate,
        :pairing_hash => job.pairing_hash,
        :master_seed => P8_MASTER_SEED,
        :apprentice_seed => job.apprentice_seed,
        :batch_order_seed => job.batch_seed,
        :regression_learning_rate => P8_LEARNING_RATE,
        :regularized_updates => job.updates,
        :post_pruning_finetune_updates => 0,
        :batch_size => training_config.batch_size,
        :validation_batch_size => training_config.validation_batch_size,
        :validation_prediction_mode => training_config.validation_prediction_mode,
        :diagnostic_teacher_forced => training_config.diagnostic_teacher_forced,
        :evaluation_interval => P8_EVALUATION_INTERVAL,
        :threshold_values => collect(P8_THRESHOLDS),
        :threshold_mode => :absolute,
        :threshold_importance_mode => :max_input_l1,
        :threshold_group_aggregation => :maximum,
        :threshold_minimum_active_groups => 1,
        :require_threshold_group_reduction => true,
        :active_inputs_definition => :global_expanded_sensor_channel_inputs,
        :pareto_objectives => (:active_inputs, :validation_matching),
        :pareto_scope => :package8_thresholds,
        :slim_evaluation_records => true,
        :result_dependent_stopping => false,
        :initial_apprentice_parameter_hash => initial_hash,
        :expert_identifier => inputs.corpus_identifier,
        :expert_path => abspath(expert_path),
        :corpus_source_files => abspath.(String.(inputs.train_dataset[:source_files])),
        :corpus_sample_count => Int(inputs.train_dataset[:sample_count]),
        :validation_source_files => abspath.(String.(inputs.validation_dataset[:source_files])),
        :validation_sample_count => Int(inputs.validation_dataset[:sample_count]),
    )
end

function save_summary!(manager, job, losses, elapsed_seconds)
    return pareto_atomic_save(
        joinpath(manager.run_directory, "summary.jld2");
        schema_version = P8_SCHEMA_VERSION,
        experiment = :package8_varying_regularizer_comparison,
        experiment_id = job.experiment_id,
        completed_at = string(Dates.now()),
        run_id = job.id,
        configuration = job.configuration,
        method = job.method,
        grouping = job.grouping,
        replicate = job.replicate,
        regularization_strength = job.regularization_strength,
        elapsed_seconds = Float64(elapsed_seconds),
        update_count = length(losses),
        final_training_loss = isempty(losses) ? NaN : last(losses),
        evaluation_count = manager.evaluation_count,
        pareto_front = manager.front,
        config_fingerprint = manager.config_fingerprint,
    )
end

function run_loaded_worker(options, directory, expert_path)
    job = options.job
    Float64(learning_rate) == P8_LEARNING_RATE || error(
        "Loaded learning rate $learning_rate does not match Package-8 value $P8_LEARNING_RATE.",
    )
    inputs = assert_varying_inputs(expert_path)
    initial_model = deepcopy(apprentice)
    initial_hash = parameter_hash(initial_model)
    training_config = ApprenticeTrainingConfig(
        regularized_updates = job.updates,
        post_pruning_finetune_updates = 0,
        batch_size = P8_BATCH_SIZE,
        proximal_interval = 1,
        reweight_interval = 10,
        regularization_strength = job.regularization_strength,
        validation_batch_size = P8_VALIDATION_BATCH_SIZE,
        validation_prediction_mode = :autoregressive,
        diagnostic_teacher_forced = false,
    )
    schedule = CandidateSchedule(
        start_update = 0,
        evaluation_interval = P8_EVALUATION_INTERVAL,
        garbage_collection_interval = P8_GARBAGE_COLLECTION_INTERVAL,
        resume_interval = P8_RESUME_INTERVAL,
    )
    config = archive_config(options, training_config, expert_path, inputs, initial_hash)
    manager = initialize_pareto_archive(directory; run_id = job.id, schedule, config)
    resume_checkpoint = load_resume_checkpoint(manager)
    previous = load_status(status_path(options.results_root, job))
    if !isnothing(previous) && Symbol(previous[:state]) === :failed && !options.retry_failed
        error("Run $(job.id) is marked failed. Use --retry-failed after repairing the cause.")
    end

    if !isnothing(resume_checkpoint) && resume_checkpoint.status === :complete
        losses = Float64.(get(resume_checkpoint.resume_state, :losses, Float64[]))
        summary_path = joinpath(directory, "summary.jld2")
        isfile(summary_path) || save_summary!(manager, job, losses, NaN)
        compact_evaluations!(manager)
        write_status!(
            status_path(options.results_root, job);
            state = :complete,
            run_id = job.id,
            experiment_id = job.experiment_id,
            configuration = job.configuration,
            replicate = job.replicate,
            update = job.updates,
            config_fingerprint = manager.config_fingerprint,
            skipped_complete = true,
            completed_at = string(Dates.now()),
        )
        println("Skipping already complete run $(job.id).")
        return manager
    end

    resume = !isnothing(resume_checkpoint) && resume_checkpoint.status in (:running, :failed)
    write_status!(
        status_path(options.results_root, job);
        state = :running,
        run_id = job.id,
        experiment_id = job.experiment_id,
        configuration = job.configuration,
        method = job.method,
        grouping = job.grouping,
        replicate = job.replicate,
        regularization_strength = job.regularization_strength,
        update = isnothing(resume_checkpoint) ? 0 : resume_checkpoint.update,
        config_fingerprint = manager.config_fingerprint,
        resumed = resume,
        started_at = string(Dates.now()),
    )

    println("Package 8 worker $(job.id)")
    println("  experiment: $(job.experiment_id)")
    println("  configuration/strength: $(job.configuration) / $(job.regularization_strength)")
    println("  replicate/seeds: $(job.replicate) / $(job.apprentice_seed), $(job.batch_seed)")
    println("  updates/batches: $(job.updates) / $P8_BATCH_SIZE, validation $P8_VALIDATION_BATCH_SIZE")
    println("  thresholds: $(join(P8_THRESHOLDS, ", ")); importance: max input L1")
    println("  resume: $resume")
    println("  output: $directory")

    started = time()
    try
        result = train_apprentice!(
            deepcopy(initial_model);
            method = job.method,
            train_dataset = inputs.train_dataset,
            validation_dataset = inputs.validation_dataset,
            config = training_config,
            archive_manager = manager,
            threshold_specs = threshold_specs(),
            group_rows_by_overlap = true,
            group_channels = job.group_channels,
            training_rng = StableRNG(job.batch_seed),
            minimum_active_groups = 1,
            threshold_importance_mode = :max_input_l1,
            threshold_minimum_active_groups = 1,
            threshold_pareto_scope = :package8_thresholds,
            require_threshold_group_reduction = true,
            resume,
        )
        elapsed = time() - started
        summary_path = save_summary!(manager, job, result.losses, elapsed)
        compact_evaluations!(manager)
        write_status!(
            status_path(options.results_root, job);
            state = :complete,
            run_id = job.id,
            experiment_id = job.experiment_id,
            configuration = job.configuration,
            method = job.method,
            grouping = job.grouping,
            replicate = job.replicate,
            regularization_strength = job.regularization_strength,
            update = job.updates,
            config_fingerprint = manager.config_fingerprint,
            initial_apprentice_parameter_hash = initial_hash,
            elapsed_seconds = elapsed,
            summary_path,
            completed_at = string(Dates.now()),
        )
        @printf("Completed %s in %.2f seconds.\n", job.id, elapsed)
        return manager
    catch exception
        write_status!(
            status_path(options.results_root, job);
            state = :failed,
            run_id = job.id,
            experiment_id = job.experiment_id,
            configuration = job.configuration,
            replicate = job.replicate,
            update = manager.last_evaluated_update,
            config_fingerprint = manager.config_fingerprint,
            error = sprint(showerror, exception),
            failed_at = string(Dates.now()),
        )
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    options = parse_worker_arguments(ARGS)
    if !isnothing(options)
        directory, expert_path = configure_worker_environment!(options)
        include(joinpath(P8_DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
        Base.invokelatest(run_loaded_worker, options, directory, expert_path)
    end
end
