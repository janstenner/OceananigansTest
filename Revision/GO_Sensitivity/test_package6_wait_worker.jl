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
