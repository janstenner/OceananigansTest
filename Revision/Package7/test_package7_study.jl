using Test

include(joinpath(@__DIR__, "Package7Study.jl"))
using .Package7Study

@testset "Package-7 deterministic job matrix" begin
    @test P7_MASTER_SEED == 20_260_829
    @test seed_plan(1) == (replicate = 1, apprentice_seed = 1_855_310_136, batch_seed = 1_941_818_438)
    @test seed_plan(2) == (replicate = 2, apprentice_seed = 1_760_770_213, batch_seed = 1_028_149_301)
    @test seed_plan(3) == (replicate = 3, apprentice_seed = 181_852_467, batch_seed = 1_377_920_448)

    jobs = study_jobs()
    @test length(jobs) == 24
    @test Set(job.configuration for job in jobs) == Set(P7_CONFIGURATION_NAMES)
    @test all(count(job -> job.configuration == name, jobs) == 3 for name in P7_CONFIGURATION_NAMES)
    @test length(unique(job.id for job in jobs)) == 24
    @test length(unique(job.relative_path for job in jobs)) == 24
    @test all(job.updates == 35_000 for job in jobs)

    single = study_jobs("go-sc")
    @test length(single) == 3
    @test all(job.regularization_strength == 0.09 for job in single)

    sweep = study_jobs("go-sc", [0.07, 0.09, 0.12])
    @test length(sweep) == 9
    @test Set(job.regularization_strength for job in sweep) == Set([0.07, 0.09, 0.12])
    @test_throws ArgumentError study_jobs("all", [0.1])
    @test_throws ArgumentError job_for("unknown", 0.1, 1)
    @test expected_evaluation_updates(100) == [0, 25, 50, 75, 100]
end

println("package7-study-tests-ok")
