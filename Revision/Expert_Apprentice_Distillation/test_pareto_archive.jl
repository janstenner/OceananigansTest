include(joinpath(@__DIR__, "ParetoArchive.jl"))

function dummy_candidate(
    threshold_id,
    active_inputs,
    matching;
    mask_value = true,
    pareto_scope = :default,
)
    group_mask = BitVector([mask_value, !mask_value])
    global_mask = BitArray(reshape(repeat(group_mask, 4), 2, 2, 2))
    return Dict{Symbol, Any}(
        :threshold_id => threshold_id,
        :validation_matching => matching,
        :active_inputs => active_inputs,
        :mask => fill(mask_value, 8),
        :group_mask => group_mask,
        :global_mask => global_mask,
        :group_importances => [1.0, 0.5],
        :active_groups => count(group_mask),
        :active_sensor_locations => count(dropdims(any(global_mask; dims = 1); dims = 1)),
        :numeric_status => :ok,
        :pareto_scope => pareto_scope,
    )
end

function run_pareto_archive_tests()
    schedule = CandidateSchedule(
        start_update = 10,
        evaluation_interval = 5,
        garbage_collection_interval = 2,
        resume_interval = 5,
    )
    @assert !should_evaluate_candidates(schedule, 5)
    @assert should_evaluate_candidates(schedule, 10)
    @assert should_evaluate_candidates(schedule, 15)
    @assert should_evaluate_candidates(schedule, 17; final = true)

    scoped_front = pareto_front([
        merge(
            dummy_candidate(:native, 100, 1.0; pareto_scope = :native),
            Dict(:candidate_id => "native"),
        ),
        merge(
            dummy_candidate(:threshold, 50, 0.5; pareto_scope = :hard_threshold),
            Dict(:candidate_id => "threshold"),
        ),
    ])
    @assert length(scoped_front) == 2

    mktempdir() do directory
        manager = initialize_pareto_archive(
            directory;
            run_id = "test-run",
            schedule,
            config = Dict(:method => :go, :seed => 1),
        )
        first_records = record_candidate_batch!(
            manager,
            10,
            [
                dummy_candidate(:native, 100, 1.0),
                dummy_candidate(:relative_01, 80, 1.2),
            ];
            model_payload = Dict(:weights => [1.0]),
        )
        @assert length(manager.front) == 2
        @assert length(unique(record[:model_path] for record in first_records)) == 1
        first_checkpoint = first_records[1][:model_path]
        @assert isfile(first_checkpoint)

        second_records = record_candidate_batch!(
            manager,
            15,
            [dummy_candidate(:native, 80, 0.9)];
            model_payload = Dict(:weights => [2.0]),
        )
        @assert length(manager.front) == 1
        @assert manager.front[1][:active_inputs] == 80
        @assert manager.front[1][:validation_matching] == 0.9
        @assert !isfile(first_checkpoint)
        @assert isfile(second_records[1][:model_path])

        third_records = record_candidate_batch!(
            manager,
            20,
            [dummy_candidate(:native, 90, 2.0)];
            model_payload = Dict(:weights => [3.0]),
        )
        @assert !third_records[1][:loadable]
        @assert isnothing(third_records[1][:model_path])
        @assert !isfile(candidate_checkpoint_path(manager, 20))
        @assert length(evaluation_files(manager)) == 3

        save_resume_checkpoint!(
            manager,
            20,
            Dict(:model => "latest", :optimizer => [1, 2, 3]),
        )
        resume = load_resume_checkpoint(manager)
        @assert resume.update == 20
        @assert resume.resume_state[:model] == "latest"

        orphan_path = candidate_checkpoint_path(manager, 25)
        pareto_atomic_save(orphan_path; model_payload = Dict(:weights => [99.0]))
        @assert isfile(orphan_path)

        reloaded = initialize_pareto_archive(
            directory;
            run_id = "test-run",
            schedule,
            config = Dict(:method => :go, :seed => 1),
        )
        @assert length(reloaded.front) == 1
        @assert reloaded.front[1][:candidate_id] == manager.front[1][:candidate_id]
        @assert !isfile(orphan_path)
        finalized = finalize_pareto_archive!(reloaded)
        @assert length(finalized.front) == 1
    end


    mktempdir() do directory
        slim_manager = initialize_pareto_archive(
            directory;
            run_id = "slim-run",
            schedule,
            config = Dict(:method => :go, :seed => 2, :slim_evaluation_records => true),
        )
        records = record_candidate_batch!(
            slim_manager,
            10,
            [
                dummy_candidate(:native, 100, 1.0),
                dummy_candidate(:relative_01, 80, 1.2; mask_value = false),
            ];
            model_payload = Dict(:weights => [1.0]),
            evaluation_metadata = Dict(:sample_count => 2),
        )
        @assert all(haskey(record, :group_mask) for record in records)
        @assert all(
            !haskey(record, key) for record in records
            for key in PARETO_SLIM_EVALUATION_OMIT_KEYS
        )
        checkpoint = candidate_checkpoint_path(slim_manager, 10)
        @assert isfile(checkpoint)
        checkpoint_metadata = JLD2.load(checkpoint, "candidate_metadata")
        @assert length(checkpoint_metadata) == 2
        hydrated = hydrate_candidate_record(records[1], directory)
        @assert hydrated[:mask] == fill(true, 8)
        @assert hydrated[:global_mask] == dummy_candidate(:native, 100, 1.0)[:global_mask]

        record_candidate_batch!(
            slim_manager,
            15,
            [dummy_candidate(:native, 80, 0.9)];
            model_payload = Dict(:weights => [2.0]),
            evaluation_metadata = Dict(:sample_count => 2),
        )
        record_candidate_batch!(
            slim_manager,
            20,
            [dummy_candidate(:native, 90, 2.0)];
            model_payload = Dict(:weights => [3.0]),
            evaluation_metadata = Dict(:sample_count => 2),
        )
        before = load_evaluation_collection(directory)
        @assert length(before.batches) == 3
        compact_path = compact_evaluations!(slim_manager)
        @assert compact_path == consolidated_evaluations_path(slim_manager)
        @assert isfile(compact_path)
        @assert isempty(evaluation_files(slim_manager))
        after = load_evaluation_collection(directory)
        @assert isequal(after.batches, before.batches)
        @assert compact_evaluations!(slim_manager) == compact_path

        # A crash after publishing the consolidated file may leave verified
        # duplicate shards. A repeated compaction removes them idempotently.
        first_batch = first(after.batches)
        mismatched_candidates = deepcopy(first_batch[:candidates])
        mismatched_candidates[1][:validation_matching] += 1.0
        pareto_atomic_save(
            evaluation_path(slim_manager, first_batch[:update]);
            schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
            run_id = slim_manager.run_id,
            update = first_batch[:update],
            candidates = mismatched_candidates,
            evaluation_metadata = after.evaluation_metadata,
            config_fingerprint = slim_manager.config_fingerprint,
            created_at = first_batch[:created_at],
        )
        mismatch_detected = false
        try
            load_evaluation_collection(directory)
        catch
            mismatch_detected = true
        end
        @assert mismatch_detected
        pareto_atomic_save(
            evaluation_path(slim_manager, first_batch[:update]);
            schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
            run_id = slim_manager.run_id,
            update = first_batch[:update],
            candidates = first_batch[:candidates],
            evaluation_metadata = after.evaluation_metadata,
            config_fingerprint = slim_manager.config_fingerprint,
            created_at = first_batch[:created_at],
        )
        @assert length(evaluation_files(slim_manager)) == 1
        compact_evaluations!(slim_manager)
        @assert isempty(evaluation_files(slim_manager))

        reloaded = initialize_pareto_archive(
            directory;
            run_id = "slim-run",
            schedule,
            config = Dict(:method => :go, :seed => 2, :slim_evaluation_records => true),
        )
        @assert reloaded.evaluation_count == 3
        @assert reloaded.last_evaluated_update == 20
    end

    mktempdir() do directory
        legacy_evaluation_directory = joinpath(directory, "evaluations")
        mkpath(legacy_evaluation_directory)
        legacy_candidate = merge(
            dummy_candidate(:native, 100, 1.0),
            Dict(
                :run_id => "legacy",
                :update => 10,
                :checkpoint_id => "checkpoint_000000000010",
                :candidate_id => candidate_identifier("legacy", 10, :native),
                :model_path => nothing,
                :loadable => false,
            ),
        )
        pareto_atomic_save(
            joinpath(legacy_evaluation_directory, "update_000000000010.jld2");
            schema_version = 1,
            run_id = "legacy",
            update = 10,
            candidates = [legacy_candidate],
            evaluation_metadata = Dict(:sample_count => 1),
            config_fingerprint = "legacy-fingerprint",
            created_at = "legacy-time",
        )
        collection = load_evaluation_collection(directory)
        @assert collection.run_id == "legacy"
        @assert only(collection.batches)[:candidates][1][:mask] == fill(true, 8)
    end
    println("pareto-archive-tests-ok")
end

run_pareto_archive_tests()
