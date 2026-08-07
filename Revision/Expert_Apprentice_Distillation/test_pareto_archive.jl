include(joinpath(@__DIR__, "ParetoArchive.jl"))

function dummy_candidate(
    threshold_id,
    active_inputs,
    matching;
    mask_value = true,
    pareto_scope = :default,
)
    return Dict{Symbol, Any}(
        :threshold_id => threshold_id,
        :validation_matching => matching,
        :active_inputs => active_inputs,
        :mask => fill(mask_value, 8),
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
    println("pareto-archive-tests-ok")
end

run_pareto_archive_tests()
