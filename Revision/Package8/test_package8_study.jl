using Test

include(joinpath(@__DIR__, "Package8Study.jl"))
using .Package8Study

@testset "Package-8 deterministic job matrix" begin
    @test P8_MASTER_SEED == 20_260_851
    @test all(all(isapprox(grid[index + 1] / grid[index], 2.5) for index in 1:2)
              for grid in values(P8_STRENGTH_GRIDS))
    @test seed_plan(1) == (replicate = 1, apprentice_seed = 1_876_371_626, batch_seed = 1_786_433_148)
    @test seed_plan(2) == (replicate = 2, apprentice_seed = 517_379_917, batch_seed = 1_720_459_459)
    @test seed_plan(3) == (replicate = 3, apprentice_seed = 221_812_090, batch_seed = 798_588_586)

    experiment_id = "260830_120000"
    jobs = study_jobs(experiment_id)
    @test length(jobs) == 72
    @test Set(job.configuration for job in jobs) == Set(P8_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name, jobs) == 9 for name in P8_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name && job.regularization_strength == strength, jobs) == 3
              for name in P8_CONFIGURATION_NAMES for strength in P8_STRENGTH_GRIDS[name])
    @test length(unique(job.id for job in jobs)) == 72
    @test length(unique(job.relative_path for job in jobs)) == 72
    @test all(job.updates == 100_000 for job in jobs)
    @test P8_BATCH_SIZE == 100
    @test P8_VALIDATION_BATCH_SIZE == 512
    @test P8_QUALITY_THRESHOLD == 3e-2
    @test all(job.experiment_id == experiment_id for job in jobs)
    @test all(first(splitpath(job.relative_path)) == experiment_id for job in jobs)

    single = study_jobs(experiment_id, "go-sc")
    @test length(single) == 9
    @test Set(job.regularization_strength for job in single) == Set([0.016, 0.04, 0.1])

    sweep = study_jobs(experiment_id, "go-sc", [0.007, 0.0175, 0.04375])
    @test length(sweep) == 9
    @test Set(job.regularization_strength for job in sweep) == Set([0.007, 0.0175, 0.04375])
    @test_throws ArgumentError study_jobs(experiment_id, "all", [0.1])
    @test_throws ArgumentError job_for(experiment_id, "unknown", 0.1, 1)
    @test_throws ArgumentError job_for("../bad", "go-sc", 0.1, 1)
    @test expected_evaluation_updates(100) == [0, 25, 50, 75, 100]
end

println("package8-study-tests-ok")
