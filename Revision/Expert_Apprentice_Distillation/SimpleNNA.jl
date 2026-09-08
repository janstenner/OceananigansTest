Base.@kwdef mutable struct SimpleNNAPolicy{M, L, O}
    mean_network::M
    logσ::L
    optimizer::O
    optimizer_state::Any = nothing
    n_actors::Int
    nna_scale::Float64
    hidden_size::Int
    target_parameter_count::Int
end

Flux.@layer SimpleNNAPolicy trainable = (mean_network, logσ)

function simple_nna_parameter_count(input_size::Integer, output_size::Integer, hidden_size::Integer;
                                    logσ_is_network::Bool = false)
    input_size > 0 || throw(ArgumentError("input_size must be positive."))
    output_size > 0 || throw(ArgumentError("output_size must be positive."))
    hidden_size > 0 || throw(ArgumentError("hidden_size must be positive."))
    mean_count =
        hidden_size * input_size + hidden_size +
        hidden_size * hidden_size + hidden_size +
        output_size * hidden_size + output_size
    logσ_count = logσ_is_network ? mean_count : output_size
    return mean_count + logσ_count
end

function closest_simple_nna_hidden_size(target::Integer, input_size::Integer, output_size::Integer;
                                        logσ_is_network::Bool = false)
    target > 0 || throw(ArgumentError("target must be positive."))
    upper = max(1, ceil(Int, sqrt(target)) + input_size + output_size)
    candidates = 1:upper
    return first(sort(collect(candidates); by = hidden -> (
        abs(simple_nna_parameter_count(input_size, output_size, hidden; logσ_is_network) - target),
        hidden,
    )))
end

function create_simple_nna_apprentice(;
    state_space,
    action_space,
    rng,
    n_actors::Integer,
    learning_rate::Real,
    betas = (0.9, 0.999),
    fun = gelu,
    tanh_end::Bool = false,
    start_logσ::Real = -10.0f0,
    target_parameter_count::Integer,
)
    input_size = size(state_space)[1]
    output_size = size(action_space)[1]
    hidden_size = closest_simple_nna_hidden_size(
        target_parameter_count,
        input_size,
        output_size;
        logσ_is_network = false,
    )
    nna_scale = hidden_size / 10
    init = Flux.glorot_uniform(rng)
    output_fun = tanh_end ? tanh : identity
    mean_network = Chain(
        Dense(input_size, hidden_size, fun; init),
        Dense(hidden_size, hidden_size, fun; init),
        Dense(hidden_size, output_size, output_fun; init),
    )
    logσ = fill(Float32(start_logσ), output_size, 1)
    optimizer = Optimisers.OptimiserChain(
        Optimisers.ClipNorm(Inf),
        Optimisers.AdamW(learning_rate, betas),
    )
    policy = SimpleNNAPolicy(
        mean_network = mean_network,
        logσ = logσ,
        optimizer = optimizer,
        n_actors = Int(n_actors),
        nna_scale = Float64(nna_scale),
        hidden_size = hidden_size,
        target_parameter_count = Int(target_parameter_count),
    )
    policy.optimizer_state = Flux.setup(optimizer, policy)
    return policy
end

function RL.prob(policy::SimpleNNAPolicy, state::AbstractArray, mask)
    means = policy.mean_network(state)
    return (; μ = means, logσ = policy.logσ)
end

simple_nna_actual_parameter_count(policy::SimpleNNAPolicy) =
    sum(length, Flux.trainables(policy); init = 0)
