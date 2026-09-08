using Flux
using IntervalSets
using JLD2
using Optimisers
using RL
using StableRNGs
using Test

include(joinpath(@__DIR__, "SimpleNNAStudy.jl"))
using .SimpleNNAStudy
include(joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation", "SimpleNNA.jl"))

parameter_count(model) = sum(length, Flux.trainables(model); init = 0)

function reference_mat_actor_parameter_count()
    action_space = Space(fill(-1..1, (1, 12)))
    state_space = Space(fill(-Inf..Inf, (360, 12)))
    agent = create_agent_mat(
        n_actors = 12,
        action_space = action_space,
        state_space = state_space,
        use_gpu = false,
        rng = StableRNG(1),
        y = 0.99f0,
        p = 0.95f0,
        learning_rate = SNN_LEARNING_RATE,
        nna_scale = 1.0,
        nna_scale_critic = 1.0,
        drop_middle_layer = true,
        drop_middle_layer_critic = true,
        fun = gelu,
        dim_model = 44,
        block_num = 1,
        head_num = 2,
        head_dim = 22,
        ffn_dim = 44,
        drop_out = 0.0,
        betas = (0.9, 0.999),
        jointPPO = false,
        customCrossAttention = true,
        positional_encoding = 3,
        start_logσ = -10.0f0,
        clip_grad = Inf,
    )
    policy = agent.policy
    return parameter_count(policy.encoder.embedding) +
           parameter_count(policy.encoder.position_encoding) +
           parameter_count(policy.encoder.ln) +
           parameter_count(policy.encoder.dropout) +
           parameter_count(policy.encoder.blocks) +
           parameter_count(policy.decoder)
end

@testset "Simple-NNA architecture and parameter budget" begin
    @test reference_mat_actor_parameter_count() == SNN_MAT_ACTOR_PARAMETER_COUNT

    action_space = Space(fill(-1..1, (1, 12)))
    state_space = Space(fill(-Inf..Inf, (360, 12)))
    policy = create_simple_nna_apprentice(
        state_space = state_space,
        action_space = action_space,
        rng = StableRNG(2),
        n_actors = 12,
        learning_rate = SNN_LEARNING_RATE,
        target_parameter_count = SNN_MAT_ACTOR_PARAMETER_COUNT,
    )
    @test policy.hidden_size == SNN_HIDDEN_SIZE
    @test policy.nna_scale == SNN_NNA_SCALE
    @test simple_nna_actual_parameter_count(policy) == SNN_PARAMETER_COUNT
    @test size(policy.mean_network[1].weight) == (102, 360)
    @test size(policy.mean_network[2].weight) == (102, 102)
    @test size(policy.mean_network[3].weight) == (1, 102)
    @test policy.mean_network[1].σ === gelu
    @test policy.mean_network[2].σ === gelu
    @test policy.mean_network[3].σ === identity

    target_difference = abs(SNN_PARAMETER_COUNT - SNN_MAT_ACTOR_PARAMETER_COUNT)
    @test target_difference < abs(
        simple_nna_parameter_count(360, 1, 103) - SNN_MAT_ACTOR_PARAMETER_COUNT,
    )
    observations = randn(Float32, 360, 12, 2)
    @test size(prob(policy, observations, nothing).μ) == (1, 12, 2)
    @test RL.prob(policy, observations, nothing).μ == prob(policy, observations, nothing).μ
    mktempdir() do directory
        path = joinpath(directory, "simple_nna_checkpoint.jld2")
        JLD2.jldsave(path; policy)
        restored = JLD2.load(path, "policy")
        @test restored isa SimpleNNAPolicy
        @test simple_nna_actual_parameter_count(restored) == SNN_PARAMETER_COUNT
        @test prob(restored, observations, nothing).μ == prob(policy, observations, nothing).μ
    end
end

println("simple-nna-architecture-tests-ok")
