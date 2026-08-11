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
    fixed_options = Package6TestWorkerWorldAgeHarness.parse_arguments([
        "--protocol", "fixed", "--parallel-test",
    ])
    @test fixed_options.parallel_test
    candidates = [Dict{Symbol, Any}(:candidate_id => "one"), Dict{Symbol, Any}(:candidate_id => "two")]
    fixed_specs = Package6TestWorkerWorldAgeHarness.parallel_episode_specs(:fixed, candidates)
    varying_specs = Package6TestWorkerWorldAgeHarness.parallel_episode_specs(:varying, candidates)
    @test length(fixed_specs) == 3
    @test length(varying_specs) == 24
    @test fixed_specs[1] == (controller_index = 0, case_index = 1)
    @test fixed_specs[end] == (controller_index = 2, case_index = 1)
    @test varying_specs[1] == (controller_index = 0, case_index = 1)
    @test varying_specs[end] == (controller_index = 2, case_index = 8)

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

    mktempdir() do output
        data = (candidates, expert_identifier = "mock_expert")
        cache = Package6TestWorkerWorldAgeHarness.cache_path(
            output,
            "expert",
            data.expert_identifier,
            nothing,
        )
        Package6TestWorkerWorldAgeHarness.save_cache(
            cache,
            episode;
            controller_id = "expert",
            expert_identifier = data.expert_identifier,
            protocol = :fixed,
            case = nothing,
        )
        Package6TestWorkerWorldAgeHarness.write_status!(
            Package6TestWorkerWorldAgeHarness.episode_status_path(output, 0, 1);
            state = :complete,
            controller_id = "expert",
            case = nothing,
        )
        @test !isnothing(Package6TestWorkerWorldAgeHarness.completed_episode_status(output, data, 0, 1))
    end
end

println("Package 6 test-worker world-age regression test passed.")
