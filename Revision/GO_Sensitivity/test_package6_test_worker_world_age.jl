using Test

module Package6TestWorkerWorldAgeHarness

include(joinpath(@__DIR__, "run_study_test_worker.jl"))

mutable struct MockEnvironment
    reward::Vector{Float64}
    state::Vector{Float32}
    steps::Int
end

function (environment::MockEnvironment)(action)
    environment.steps += 1
    environment.reward .= sum(action)
    return environment
end

module MockRL

reset!(environment) = (environment.steps = 0; environment)

end

function install_mock_runtime!()
    Base.eval(@__MODULE__, quote
        const RL = MockRL
        const env = MockEnvironment([0.0], zeros(Float32, 12), 0)
        const CORPUS = Dict(
            :test => Dict(11 => nothing, 22 => nothing),
        )
        generate_random_init(; kwargs...) = nothing
        state_Nu(environment) = environment.steps
    end)
    return nothing
end

end

using .Package6TestWorkerWorldAgeHarness

@testset "Package 6 dynamically loaded test runtime" begin
    Package6TestWorkerWorldAgeHarness.install_mock_runtime!()
    episode = Package6TestWorkerWorldAgeHarness.run_episode(
        :fixed,
        nothing,
        () -> zeros(Float32, 12),
    )
    @test length(episode.rewards) == Package6TestWorkerWorldAgeHarness.TEST_STEPS
    @test episode.global_nusselt == collect(1.0:Package6TestWorkerWorldAgeHarness.TEST_STEPS)
    @test size(episode.actions) == (Package6TestWorkerWorldAgeHarness.TEST_STEPS, 12)
    @test length(Package6TestWorkerWorldAgeHarness.test_cases(:varying)) == 8
end

println("Package 6 test-worker world-age regression test passed.")
