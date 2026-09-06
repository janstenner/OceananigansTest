using Test, JLD2
include("MaskedStudy.jl")
const S = MaskedStudy

@testset "Frozen candidate selection and result identity" begin
    mktempdir() do root
        for grouping in (:gc, :sc), (i, method) in enumerate(("go", "gr", "group-lasso", "growl"))
            configuration = "$method-$grouping"
            directory = joinpath(root, configuration)
            mkpath(joinpath(directory, "analysis"))
            checkpoint = joinpath(directory, "checkpoint.jld2")
            JLD2.jldsave(checkpoint; unused_model = 1)
            candidate = (; configuration, candidate_id = configuration, mask = vcat(ones(Float32, 3), zeros(Float32, 357)),
                active_inputs = i <= 2 ? 36 : 72, active_groups = i <= 2 ? 1 : 2,
                validation_matching = i == 2 ? 0.002 : 0.004)
            JLD2.jldsave(joinpath(directory, "analysis", "selected_test_candidate.jld2");
                experiment = :package7_fixed_regularizer_comparison, frozen_before_test = true,
                selection_uses_test_data = false, candidate, checkpoint_path = checkpoint,
                checkpoint_sha256 = S.file_sha256(checkpoint))
        end
        @test S.select_candidate(:fixed, :gc, root).configuration == "gr-gc"
        @test S.select_candidate(:fixed, :sc, root).configuration == "gr-sc"
        @test_throws ErrorException S.select_candidate(:varying, :gc, root)
        path = joinpath(root, "result.jld2")
        @test !S.complete_result(path, "a", 2000)
        JLD2.jldsave(path; manifest_identity = "a", episode_target = 2000, episodes_completed = 2000, status = "complete")
        @test S.complete_result(path, "a", 2000)
        @test_throws ErrorException S.complete_result(path, "b", 2000)
        @test_throws ErrorException S.complete_result(path, "a", 4000)
    end
end
