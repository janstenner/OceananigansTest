using Test

include(joinpath(@__DIR__, "SimpleNNAStudy.jl"))
using .SimpleNNAStudy

@testset "Simple-NNA deterministic job matrix" begin
    @test SNN_MASTER_SEED == 20_260_851
    @test SNN_APPRENTICE_ARCHITECTURE === :simple_nna
    @test SNN_NNA_SCALE == 10.2
    @test SNN_HIDDEN_SIZE == 102
    @test SNN_MAT_ACTOR_PARAMETER_COUNT == 47_698
    @test SNN_PARAMETER_COUNT == 47_432
    @test all(all(isapprox(grid[index + 1] / grid[index], 2.5) for index in 1:2)
              for grid in values(SNN_STRENGTH_GRIDS))
    @test seed_plan(1) == (replicate = 1, apprentice_seed = 1_876_371_626, batch_seed = 1_786_433_148)
    @test seed_plan(2) == (replicate = 2, apprentice_seed = 517_379_917, batch_seed = 1_720_459_459)
    @test seed_plan(3) == (replicate = 3, apprentice_seed = 221_812_090, batch_seed = 798_588_586)

    experiment_id = "260830_120000"
    jobs = study_jobs(experiment_id)
    @test length(jobs) == 36
    @test Set(job.configuration for job in jobs) == Set(SNN_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name, jobs) == 9 for name in SNN_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name && job.regularization_strength == strength, jobs) == 3
              for name in SNN_CONFIGURATION_NAMES for strength in SNN_STRENGTH_GRIDS[name])
    @test length(unique(job.id for job in jobs)) == 36
    @test length(unique(job.relative_path for job in jobs)) == 36
    @test all(job.updates == 100_000 for job in jobs)
    @test SNN_BATCH_SIZE == 100
    @test SNN_VALIDATION_BATCH_SIZE == 512
    @test SNN_QUALITY_THRESHOLD == 3e-2
    @test resolved_thresholds() == collect(SNN_THRESHOLDS)
    @test resolved_thresholds([0.08, 0.02, 0.04, 0.04]) == [0.0, 0.02, 0.04, 0.08]
    @test_throws ArgumentError resolved_thresholds([0.0, 0.02])
    @test_throws ArgumentError resolved_thresholds([-0.01])
    @test all(job.experiment_id == experiment_id for job in jobs)
    @test all(first(splitpath(job.relative_path)) == experiment_id for job in jobs)

    single = study_jobs(experiment_id, "go-sc")
    @test length(single) == 9
    @test Set(job.regularization_strength for job in single) == Set([0.016, 0.04, 0.1])

    sweep = study_jobs(experiment_id, "go-sc", [0.007, 0.0175, 0.04375])
    @test length(sweep) == 9
    @test Set(job.regularization_strength for job in sweep) == Set([0.007, 0.0175, 0.04375])
    @test_throws ArgumentError study_jobs(experiment_id, "all", [0.1])
    @test_throws ArgumentError job_for(experiment_id, "group-lasso-gc", 0.1, 1)
    @test_throws ArgumentError job_for("../bad", "go-sc", 0.1, 1)
    @test expected_evaluation_updates(100) == [0, 25, 50, 75, 100]
end

println("simple_nna-study-tests-ok")
