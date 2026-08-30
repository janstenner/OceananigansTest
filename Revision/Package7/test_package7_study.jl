using Test

include(joinpath(@__DIR__, "Package7Study.jl"))
using .Package7Study

@testset "Package-7 deterministic job matrix" begin
    @test P7_MASTER_SEED == 20_260_850
    @test all(all(isapprox(grid[index + 1] / grid[index], 2.5) for index in 1:2)
              for grid in values(P7_STRENGTH_GRIDS))
    @test seed_plan(1) == (replicate = 1, apprentice_seed = 996_898_248, batch_seed = 1_207_818_757)
    @test seed_plan(2) == (replicate = 2, apprentice_seed = 1_452_413_696, batch_seed = 1_103_313_457)
    @test seed_plan(3) == (replicate = 3, apprentice_seed = 497_948_374, batch_seed = 1_844_296_950)

    experiment_id = "260830_120000"
    jobs = study_jobs(experiment_id)
    @test length(jobs) == 72
    @test Set(job.configuration for job in jobs) == Set(P7_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name, jobs) == 9 for name in P7_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name && job.regularization_strength == strength, jobs) == 3
              for name in P7_CONFIGURATION_NAMES for strength in P7_STRENGTH_GRIDS[name])
    @test length(unique(job.id for job in jobs)) == 72
    @test length(unique(job.relative_path for job in jobs)) == 72
    @test all(job.updates == 35_000 for job in jobs)
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

println("package7-study-tests-ok")
