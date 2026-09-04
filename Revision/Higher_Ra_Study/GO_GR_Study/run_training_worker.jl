using Dates
using JLD2
using Printf
using SHA
using StableRNGs

include(joinpath(@__DIR__, "HigherRaGOStudy.jl"))
using .HigherRaGOStudy

const DISTILLATION_DIRECTORY = normpath(joinpath(@__DIR__, "..", "..", "Expert_Apprentice_Distillation"))

function worker_usage(io::IO = stdout)
    println(io, """
    Higher-Ra GO/GR training worker.

    Usage:
      julia --startup-file=no --project=. run_training_worker.jl \
        --study ra5e4|ra1e5 --experiment-id ID --config go-gc|go-sc|gr-gc|gr-sc \
        --strength VALUE --replicate 1|2|3 [--threshold VALUE ...] \
        [--results-dir PATH] [--retry-failed] [--smoke-updates N]
    """)
end

function parse_worker_arguments(arguments)
    values = Dict{String, Any}(
        "study" => nothing,
        "experiment_id" => nothing,
        "config" => nothing,
        "strength" => nothing,
        "replicate" => nothing,
        "results_dir" => DEFAULT_RESULTS_ROOT,
        "retry_failed" => false,
        "smoke_updates" => nothing,
        "thresholds" => Float64[],
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
        elseif argument == "--threshold"
            index < length(arguments) || error("Missing value after $argument.")
            push!(values["thresholds"], parse(Float64, arguments[index + 1]))
            index += 2
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
    for key in ("study", "experiment_id", "config", "strength", "replicate")
        isnothing(values[key]) && error("--$(replace(key, "_" => "-")) is required.")
    end
    study_tag = normalize_study(values["study"])
    strength = parse(Float64, string(values["strength"]))
    replicate = parse(Int, string(values["replicate"]))
    smoke_updates = isnothing(values["smoke_updates"]) ? nothing :
        parse(Int, string(values["smoke_updates"]))
    !isnothing(smoke_updates) && smoke_updates < 0 && error(
        "--smoke-updates must be nonnegative.",
    )
    updates = isnothing(smoke_updates) ? HR_UPDATES : smoke_updates
    return (;
        study = study_tag,
        job = job_for(study_tag, values["experiment_id"], values["config"],
                      strength, replicate; updates),
        results_root = abspath(string(values["results_dir"])),
        retry_failed = Bool(values["retry_failed"]),
        smoke_updates,
        thresholds = resolved_mask_thresholds(values["thresholds"]),
    )
end

function configure_worker_environment!(options)
    job = options.job
    study_config = study(options.study)
    directory = run_directory(options.results_root, job)
    ENV["DISTILLATION_PROTOCOL"] = "varying"
    ENV["DISTILLATION_AUTOLOAD_PROTOCOL"] = "varying"
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "false"
    ENV["DISTILLATION_GROUP_CHANNELS"] = string(job.group_channels)
    ENV["DISTILLATION_RUN_FILE"] = study_config.run_file
    ENV["REVISION_RUN_SEED"] = string(job.apprentice_seed)
    ENV["DISTILLATION_VARYING_EXPERT_PATH"] = study_config.expert
    ENV["DISTILLATION_WORKER_DIRECTORY"] = joinpath(study_config.distillation_root, "varying")
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(directory, "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(directory, "apprentice_output")
    return (directory, study_config)
end

function metadata_value(mapping, key::Symbol)
    haskey(mapping, key) && return mapping[key]
    haskey(mapping, string(key)) && return mapping[string(key)]
    error("Missing '$key' in Higher-Ra corpus metadata.")
end

function assert_higher_ra_inputs(study_config)
    isfile(study_config.expert) || error("Higher-Ra expert is missing: $(study_config.expert)")
    train_coverage = distillation_coverage(:varying, :train)
    train_coverage.complete || error(
        "Incomplete $(study_config.tag) train corpus: expected $(train_coverage.expected), found $(train_coverage.actual).",
    )
    validation_coverage = distillation_coverage(:varying, :validation)
    validation_coverage.complete || error(
        "Incomplete $(study_config.tag) validation corpus: expected $(validation_coverage.expected), found $(validation_coverage.actual).",
    )
    test_coverage = distillation_coverage(:varying, :test)
    test_coverage.complete || error(
        "Incomplete $(study_config.tag) test corpus: expected $(test_coverage.expected), found $(test_coverage.actual).",
    )
    train_dataset = distillation_dataset(:varying, :train)
    validation_dataset = distillation_dataset(:varying, :validation)
    test_dataset = distillation_dataset(:varying, :test)
    length(train_dataset[:source_files]) == 40 || error("Expected 40 train shards.")
    length(validation_dataset[:source_files]) == 2 || error("Expected 2 validation shards.")
    length(test_dataset[:source_files]) == 4 || error("Expected 4 test shards.")
    train_dataset === validation_dataset && error("Training and validation must be distinct objects.")
    identifiers = Set(string(dataset[:expert_identifier]) for dataset in
                      (train_dataset, validation_dataset, test_dataset))
    length(identifiers) == 1 || error("Higher-Ra corpus splits use different experts.")
    corpus_identifier = only(identifiers)
    loaded_identifier = string(DISTILLATION_EXPERT_METADATA[:identifier])
    corpus_identifier == loaded_identifier || error(
        "Corpus expert $corpus_identifier does not match loaded expert $loaded_identifier.",
    )
    expected_identifier = "sha256:$(file_sha256(study_config.expert))"
    loaded_identifier == expected_identifier || error(
        "Loaded expert identity does not match $(study_config.expert).",
    )
    for dataset in (train_dataset, validation_dataset, test_dataset)
        metadata = dataset[:observation_metadata]
        Symbol(metadata_value(metadata, :higher_ra_study)) == study_config.tag || error(
            "Corpus metadata belongs to a different Higher-Ra study.",
        )
        Float64(metadata_value(metadata, :rayleigh)) == study_config.rayleigh || error(
            "Corpus metadata has the wrong Rayleigh number.",
        )
    end
    return (; train_coverage, validation_coverage, test_coverage, train_dataset,
            validation_dataset, test_dataset, corpus_identifier)
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

function threshold_specs(thresholds)
    return [
        HardThresholdSpec(
            Symbol("threshold_", replace(string(value), "." => "p")),
            :absolute,
            value;
            analysis_scope = :higher_ra_go_gr,
        )
        for value in thresholds[2:end]
    ]
end

function archive_config(options, training_config, study_config, inputs, initial_hash)
    job = options.job
    sources = source_manifest(study_config.tag)
    return Dict{Symbol, Any}(
        :schema_version => HR_SCHEMA_VERSION,
        :experiment => :higher_ra_go_gr_comparison,
        :experiment_id => job.experiment_id,
        :scientific_scope => isnothing(options.smoke_updates) ? :higher_ra_go_gr : :smoke_test,
        :study => study_config.tag,
        :protocol => study_config.protocol,
        :distillation_protocol => :varying,
        :rayleigh => study_config.rayleigh,
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
        :master_seed => HR_MASTER_SEED,
        :apprentice_seed => job.apprentice_seed,
        :batch_order_seed => job.batch_seed,
        :regression_learning_rate => HR_LEARNING_RATE,
        :regularized_updates => job.updates,
        :post_pruning_finetune_updates => 0,
        :batch_size => training_config.batch_size,
        :validation_batch_size => training_config.validation_batch_size,
        :validation_prediction_mode => training_config.validation_prediction_mode,
        :diagnostic_teacher_forced => training_config.diagnostic_teacher_forced,
        :evaluation_interval => HR_EVALUATION_INTERVAL,
        :mask_threshold_values => copy(options.thresholds),
        :quality_threshold_values => collect(HR_QUALITY_THRESHOLDS),
        :threshold_mode => :absolute,
        :threshold_importance_mode => :max_input_l1,
        :threshold_group_aggregation => :maximum,
        :threshold_minimum_active_groups => 1,
        :require_threshold_group_reduction => true,
        :active_inputs_definition => :global_expanded_sensor_channel_inputs,
        :pareto_objectives => (:active_inputs, :validation_matching),
        :pareto_scope => :higher_ra_go_gr_thresholds,
        :slim_evaluation_records => true,
        :result_dependent_stopping => false,
        :initial_apprentice_parameter_hash => initial_hash,
        :expert_identifier => inputs.corpus_identifier,
        :expert_path => sources.expert_path,
        :expert_sha256 => sources.expert_sha256,
        :run_file_path => sources.run_file_path,
        :run_file_sha256 => sources.run_file_sha256,
        :state_corpus_path => sources.state_corpus_path,
        :state_corpus_sha256 => sources.state_corpus_sha256,
        :distillation_root => sources.distillation_root,
        :corpus_source_files => abspath.(String.(inputs.train_dataset[:source_files])),
        :corpus_sample_count => Int(inputs.train_dataset[:sample_count]),
        :validation_source_files => abspath.(String.(inputs.validation_dataset[:source_files])),
        :validation_sample_count => Int(inputs.validation_dataset[:sample_count]),
        :test_source_files => abspath.(String.(inputs.test_dataset[:source_files])),
        :test_sample_count => Int(inputs.test_dataset[:sample_count]),
        :expert_baseline_path => sources.expert_baseline_path,
        :expert_baseline_sha256 => sources.expert_baseline_sha256,
        :unactuated_baseline_path => sources.unactuated_baseline_path,
        :unactuated_baseline_sha256 => sources.unactuated_baseline_sha256,
    )
end

function save_summary!(manager, job, losses, elapsed_seconds)
    return pareto_atomic_save(
        joinpath(manager.run_directory, "summary.jld2");
        schema_version = HR_SCHEMA_VERSION,
        experiment = :higher_ra_go_gr_comparison,
        experiment_id = job.experiment_id,
        study = job.study,
        rayleigh = job.rayleigh,
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

function run_loaded_worker(options, directory, study_config)
    job = options.job
    Float64(RA) == study_config.rayleigh || error(
        "Loaded run file has Ra=$(Float64(RA)); expected $(study_config.rayleigh).",
    )
    Float64(learning_rate) == HR_LEARNING_RATE || error(
        "Loaded learning rate $learning_rate does not match $HR_LEARNING_RATE.",
    )
    inputs = assert_higher_ra_inputs(study_config)
    initial_model = deepcopy(apprentice)
    initial_hash = parameter_hash(initial_model)
    training_config = ApprenticeTrainingConfig(
        regularized_updates = job.updates,
        post_pruning_finetune_updates = 0,
        batch_size = HR_BATCH_SIZE,
        proximal_interval = 1,
        reweight_interval = 10,
        regularization_strength = job.regularization_strength,
        validation_batch_size = HR_VALIDATION_BATCH_SIZE,
        validation_prediction_mode = :autoregressive,
        diagnostic_teacher_forced = false,
    )
    schedule = CandidateSchedule(
        start_update = 0,
        evaluation_interval = HR_EVALUATION_INTERVAL,
        garbage_collection_interval = HR_GARBAGE_COLLECTION_INTERVAL,
        resume_interval = HR_RESUME_INTERVAL,
    )
    config = archive_config(options, training_config, study_config, inputs, initial_hash)
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
            study = job.study,
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
        study = job.study,
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

    println("Higher-Ra worker $(job.id)")
    println("  study: $(study_config.label)")
    println("  configuration/strength: $(job.configuration) / $(job.regularization_strength)")
    println("  replicate/seeds: $(job.replicate) / $(job.apprentice_seed), $(job.batch_seed)")
    println("  updates/batches: $(job.updates) / $HR_BATCH_SIZE, validation $HR_VALIDATION_BATCH_SIZE")
    println("  mask thresholds: $(join(options.thresholds, ", ")); importance: max input L1")
    println("  quality thresholds: $(join(HR_QUALITY_THRESHOLDS, ", "))")
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
            threshold_specs = threshold_specs(options.thresholds),
            group_rows_by_overlap = true,
            group_channels = job.group_channels,
            training_rng = StableRNG(job.batch_seed),
            minimum_active_groups = 1,
            threshold_importance_mode = :max_input_l1,
            threshold_minimum_active_groups = 1,
            threshold_pareto_scope = :higher_ra_go_gr_thresholds,
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
            study = job.study,
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
            study = job.study,
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
        directory, study_config = configure_worker_environment!(options)
        include(joinpath(DISTILLATION_DIRECTORY, "Expert_Apprentice.jl"))
        Base.invokelatest(run_loaded_worker, options, directory, study_config)
    end
end
