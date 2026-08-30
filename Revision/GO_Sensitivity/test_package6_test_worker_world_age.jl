using JLD2
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

function install_and_run_mock_episode()
    install_mock_runtime!()
    return run_episode(:fixed, nothing, () -> zeros(Float32, 12))
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

    # This deliberately loads RL/env and consumes them in the same precompiled
    # caller, matching the Julia-1.12 failure mode of the real test worker.
    episode = Package6TestWorkerWorldAgeHarness.install_and_run_mock_episode()
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


    mktempdir() do baseline_root
        previous = get(ENV, "REVISION_BASELINE_RESULTS_DIR", nothing)
        try
            ENV["REVISION_BASELINE_RESULTS_DIR"] = baseline_root
            path = joinpath(baseline_root, "fixed", "expert.jld2")
            mkpath(dirname(path))
            baseline_episode = (
                case_id = "fixed_shared",
                choice = nothing,
                rewards = fill(-2.0, Package6TestWorkerWorldAgeHarness.TEST_STEPS),
                state_nusselt = collect(1.0:Package6TestWorkerWorldAgeHarness.TEST_STEPS),
                actions = zeros(Float32, Package6TestWorkerWorldAgeHarness.TEST_STEPS, 12),
            )
            JLD2.jldsave(
                path;
                status = :complete,
                protocol = :fixed,
                controller = :expert,
                steps = Package6TestWorkerWorldAgeHarness.TEST_STEPS,
                case_count = 1,
                expert_sha256 = "abc123",
                episodes = [baseline_episode],
            )
            loaded = Package6TestWorkerWorldAgeHarness.baseline_expert_episodes(
                :fixed,
                [nothing],
                "sha256:abc123",
            )
            @test !isnothing(loaded)
            @test loaded.episodes["fixed_shared"].global_nusselt == baseline_episode.state_nusselt
            @test_throws ErrorException Package6TestWorkerWorldAgeHarness.baseline_expert_episodes(
                :fixed,
                [nothing],
                "sha256:different",
            )
        finally
            if isnothing(previous)
                delete!(ENV, "REVISION_BASELINE_RESULTS_DIR")
            else
                ENV["REVISION_BASELINE_RESULTS_DIR"] = previous
            end
        end
    end
end

println("Package 6 test-worker world-age regression test passed.")
