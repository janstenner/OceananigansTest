using Test

include(joinpath(@__DIR__, "analyze_study_worker.jl"))

function synthetic_record(job, groups, mse, update)
    mask = falses(3, 4)
    mask[1:min(length(mask), max(1, groups % length(mask) + 1))] .= true
    return Dict{Symbol, Any}(
        :active_groups => groups,
        :active_inputs => count(mask),
        :active_sensor_locations => count(mask),
        :validation_matching => mse,
        :update => update,
        :candidate_id => "$(job.id)_$update",
        :run_id => job.id,
        :numeric_status => :ok,
        :threshold_id => :native,
        :global_mask => mask,
        :mask => BitVector(vec(mask)),
        :protocol => job.protocol,
        :method => job.method,
        :strength_index => job.strength_index,
        :regularization_strength => job.regularization_strength,
        :replicate => job.replicate,
        :source_run_directory => "synthetic",
    )
end

@testset "Package 6 aggregate metrics and plot smoke" begin
    jobs = study_jobs(:fixed)
    runs = [begin
        offset = 0.01 * job.strength_index + 0.001 * job.replicate
        records = [
            synthetic_record(job, 96, 0.10 + offset, 0),
            synthetic_record(job, 48, 0.03 + offset, 25),
            synthetic_record(job, 24, 0.009 + offset / 10, 50),
            synthetic_record(job, 12, 0.006 + offset / 10, 75),
        ]
        (
            job,
            records,
            summary = Dict{String, Any}("elapsed_seconds" => 1.0),
            config = Dict{Symbol, Any}(:expert_path => "expert.jld2"),
        )
    end for job in jobs]
    audit = (runs, expert_identifier = "synthetic", audit_passed = true)
    metrics = build_metrics(audit)
    @test length(metrics.run_summary_rows) == 18
    @test length(metrics.attainment_rows) == 6 * 97 * 3
    mktempdir() do directory
        options = (protocol = :fixed, results_root = directory, poll_seconds = 0, timeout_seconds = 0, skip_test = true)
        metrics_path = persist_metrics(options, audit, metrics)
        persisted_metrics = JLD2.load(metrics_path)
        @test !haskey(persisted_metrics, "checkpoints")
        @test haskey(persisted_metrics, "checkpoint_rows")
        @test length(persisted_metrics["checkpoint_rows"]) == sum(length, values(metrics.checkpoints))
        @test all(
            !haskey(row, :global_mask) && !haskey(row, :mask)
            for row in persisted_metrics["checkpoint_rows"]
        )
        paths = make_plots(options, audit, metrics)
        manifest = joinpath(directory, "fixed", "analysis", "candidate_manifest.jld2")
        selected = copy(first(first(runs).records))
        selected[:selection_role] = :C_match
        atomic_save(
            manifest;
            schema_version = P6_SCHEMA_VERSION,
            protocol = :fixed,
            frozen_before_test = true,
            candidates = [selected],
        )
        report = write_report(options, audit, metrics, paths, manifest, nothing)
        @test isfile(joinpath(directory, "fixed", "analysis", "metrics.jld2"))
        @test isfile(paths[:pareto])
        @test filesize(paths[:pareto]) > 0
        @test isfile(paths[:interactive_3d])
        @test occursin("plotly", lowercase(read(paths[:interactive_3d], String)))
        @test occursin("Stability definitions", read(report, String))
    end
end

println("Package 6 plot smoke test passed.")
