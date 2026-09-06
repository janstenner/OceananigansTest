using Test, RL
include("MaskedTraining.jl")

mutable struct MockEnv <: RL.AbstractEnv
    state::Matrix{Float32}
    reward::Vector{Float64}
    steps::Int
    full_steps::Vector{Matrix{Float32}}
end
RL.reset!(e::MockEnv) = (e.state = Float32[2 3; 5 7]; e.steps = 0)
RL.is_terminated(e::MockEnv) = e.steps == 2
RL.is_truncated(e::MockEnv) = false
function (e::MockEnv)(action)
    push!(e.full_steps, copy(e.state))
    @test all(e.state .> 0)
    e.reward = [sum(e.state)] # Detect any accidental reward masking.
    e.state = e.state .+ 1
    e.steps += 1
end
mutable struct MockAgent
    seen::Vector{Matrix{Float32}}
    next_seen::Vector{Matrix{Float32}}
    rewards::Vector{Float64}
end
function (a::MockAgent)(e::MockEnv)
    @test all(iszero, e.state[2, :])
    zeros(2)
end
function (a::MockAgent)(stage::RL.AbstractStage, e::MockEnv, args...)
    @test all(iszero, e.state[2, :])
    stage === PRE_ACT_STAGE && push!(a.seen, copy(e.state))
    if stage === POST_ACT_STAGE
        push!(a.next_seen, copy(e.state))
        append!(a.rewards, e.reward)
    end
end
mutable struct MockHook
    rewards::Vector{Float64}
    total::Float64
end
function (h::MockHook)(stage::RL.AbstractStage, a, e, args...)
    @test all(e.state .> 0)
    stage === POST_ACT_STAGE && (h.total += only(e.reward))
    if stage === POST_EPISODE_STAGE
        push!(h.rewards, h.total)
        h.total = 0
    end
end

@testset "Masked policy and bootstrap observations; full rewards and hooks" begin
    env = MockEnv(Float32[2 3; 5 7], [0.0], 0, Matrix{Float32}[])
    agent = MockAgent(Matrix{Float32}[], Matrix{Float32}[], Float64[])
    hook = MockHook(Float64[], 0.0)
    @test train_masked!(agent, hook, env, Float32[1, 0], 2; progress_every = 0) == 4
    @test hook.rewards == [38, 38]
    @test agent.rewards == [17, 21, 17, 21]
    @test agent.seen[1] == Float32[2 3; 0 0]
    @test agent.next_seen[2] == Float32[4 5; 0 0]
    @test length(agent.seen) == length(agent.next_seen) == 4
    full = env.state
    @test_throws ErrorException with_masked_state(() -> error("callback failure"), env, Float32[1, 0])
    @test env.state === full
    @test train_masked!(agent, hook, env, Float32[1, 0], 0) == 0
end
