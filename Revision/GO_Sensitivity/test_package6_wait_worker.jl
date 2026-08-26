using SHA
using Test

include(joinpath(@__DIR__, "analyze_study_worker.jl"))

options_for(root; timeout = 0) = (
    protocol = :fixed,
    results_root = root,
    poll_seconds = 0,
    timeout_seconds = timeout,
    skip_test = true,
)

function slim_test_candidate(groups, mse)
    group_mask = BitVector(vcat(fill(true, groups), fill(false, 4 - groups)))
    global_mask = BitArray(reshape(repeat(group_mask, 3), 3, 4, 1))
    return Dict{Symbol, Any}(
        :threshold_id => :native,
        :validation_matching => mse,
        :active_inputs => count(global_mask),
        :mask => Float32.(repeat(group_mask, 3)),
        :group_mask => group_mask,
        :global_mask => global_mask,
        :group_importances => collect(4.0:-1.0:1.0),
        :active_groups => count(group_mask),
        :active_sensor_locations => count(dropdims(any(global_mask; dims = 1); dims = 1)),
        :numeric_status => :ok,
        :pareto_scope => :native,
        :method => :go,
        :regularization_strength => 0.0015,
    )
end

@testset "Analysis wait worker states and timeout" begin
    mktempdir() do directory
        options = options_for(directory)
        jobs = study_jobs(:fixed)
        @test_throws Exception wait_for_runs(options)
        @test isfile(joinpath(directory, "fixed", "analysis", "failure_report.md"))

        for job in jobs
            write_status!(status_path(directory, job); state = :running, run_id = job.id)
        end
        @test_throws Exception wait_for_runs(options)

        write_status!(status_path(directory, first(jobs)); state = :failed, run_id = first(jobs).id, error_message = "synthetic failure")
        @test_throws Exception wait_for_runs(options_for(directory; timeout = 10))
        @test occursin("synthetic failure", read(joinpath(directory, "fixed", "analysis", "failure_report.md"), String))

        for job in jobs
            write_status!(status_path(directory, job); state = :complete, run_id = job.id)
        end
        @test length(wait_for_runs(options)) == 18
    end
end


@testset "Analysis loads shards and consolidated evaluations identically" begin
    mktempdir() do directory
        original_job = first(study_jobs(:fixed))
        job = merge(original_job, (updates = 25,))
        run_path = run_directory(directory, job)
        schedule = CandidateSchedule(
            start_update = 0,
            evaluation_interval = 25,
            garbage_collection_interval = 5,
            resume_interval = 25,
        )
        manager = initialize_pareto_archive(
            run_path;
            run_id = job.id,
            schedule,
            config = Dict(:experiment => :test, :slim_evaluation_records => true),
        )
        for (update, groups, mse) in ((0, 4, 0.1), (25, 2, 0.01))
            record_candidate_batch!(
                manager,
                update,
                [slim_test_candidate(groups, mse)];
                model_payload = Dict(:update => update),
                evaluation_metadata = Dict(:sample_count => 2),
            )
        end
        pareto_atomic_save(
            joinpath(run_path, "summary.jld2");
            config_fingerprint = manager.config_fingerprint,
            elapsed_seconds = 1.0,
        )
        write_status!(
            status_path(directory, job);
            state = :complete,
            run_id = job.id,
            config_fingerprint = manager.config_fingerprint,
            update = job.updates,
        )
        options = options_for(directory)
        shard_run = load_run(options, job)
        @test length(shard_run.records) == 2
        @test all(!haskey(record, :mask) for record in shard_run.records)

        compact_evaluations!(manager)
        consolidated_run = load_run(options, job)
        @test isequal(consolidated_run.records, shard_run.records)
        @test isfile(consolidated_evaluations_path(manager))
        @test isempty(evaluation_files(manager))
        hydrated = hydrate_candidate_record(
            last(consolidated_run.records),
            run_path;
            required_keys = (:mask, :global_mask),
        )
        @test haskey(hydrated, :mask)
        @test haskey(hydrated, :global_mask)
    end
end

@testset "Candidate manifest freezes before test and is immutable" begin
    mktempdir() do directory
        options = options_for(directory)
        checkpoint = joinpath(directory, "checkpoint.jld2")
        write(checkpoint, "synthetic checkpoint marker")
        candidate = Dict{Symbol, Any}(
            :active_groups => 2,
            :validation_matching => 0.005,
            :update => 100,
            :candidate_id => "selected",
            :run_id => "p6_f_go_s01_r01",
            :numeric_status => :ok,
            :model_path => checkpoint,
            :source_run_directory => directory,
            :regularization_strength => 0.0015,
        )
        run = (
            job = (method = :go,),
            records = [candidate],
            config = Dict{Symbol, Any}(:expert_path => "server/expert.jld2"),
        )
        audit = (runs = [run], expert_identifier = "expert-hash")
        manifest = freeze_candidate_manifest(options, audit, nothing)
        @test isfile(manifest)
        first_hash = bytes2hex(SHA.sha256(read(manifest)))
        @test freeze_candidate_manifest(options, audit, nothing) == manifest
        @test bytes2hex(SHA.sha256(read(manifest))) == first_hash

        changed = copy(candidate)
        changed[:candidate_id] = "changed"
        changed[:validation_matching] = 0.004
        changed_audit = (runs = [(job = (method = :go,), records = [changed], config = run.config)], expert_identifier = "expert-hash")
        @test_throws Exception freeze_candidate_manifest(options, changed_audit, nothing)
        @test bytes2hex(SHA.sha256(read(manifest))) == first_hash
    end
end

println("Package 6 wait/manifest tests passed.")
