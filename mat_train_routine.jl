p = agent.policy
t = agent.trajectory






rng = p.rng
γ = p.γ
λ = p.λ
n_epochs = p.n_epochs
n_microbatches = p.n_microbatches
clip_range = p.clip_range
w₁ = p.actor_loss_weight
w₂ = p.critic_loss_weight
w₃ = p.entropy_loss_weight
D = RL.device(p.encoder)
to_device(x) = send_to_device(D, x)

n_actors, n_rollout = size(t[:terminal])
@assert n_rollout % n_microbatches == 0 "size mismatch"
microbatch_size = n_rollout ÷ n_microbatches

n = length(t)
states = to_device(t[:state])

values = reshape(flatten_batch(t[:values]), n_actors, :)
next_values = cat(values[:,2:end], p.next_values, dims=2)

rewards = t[:reward]
terminal = t[:terminal]

if p.jointPPO
    values = values[1,:]
    next_values = next_values[1,:]
    rewards = rewards[1,:]
    terminal = terminal[1,:]
end





# dims = 1

# advantages = similar(rewards, promote_type(eltype(rewards), Float32))

# for (pr′, pr, pv, pnv, pt) in zip(
#     eachslice(advantages, dims = dims),
#     eachslice(rewards, dims = dims),
#     eachslice(values, dims = dims),
#     eachslice(next_values, dims = dims),
#     eachslice(terminal, dims = dims),
# )
#     #_generalized_advantage_estimation!(r′, r, v, nv, γ, λ, t)
#     global r′, r, v, nv, t
#     r′ = pr′
#     r = pr
#     v = pv
#     nv = pnv
#     t = pt
#     break
# end

# gae = 0.0f0

# for i in length(r):-1:1
#     is_continue = isnothing(t) ? true : (!t[i])
#     delta = r[i] + γ * nv[i] * is_continue - v[i]
#     gae = delta + γ * λ * is_continue * gae
#     r′[i] = gae
# end





advantages = generalized_advantage_estimation(
    rewards,
    values,
    next_values,
    γ,
    λ;
    dims=2,
    terminal=terminal
)

if p.jointPPO
    returns = to_device(advantages .+ values[1:n_rollout])
else
    returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
end
advantages = to_device(advantages)

actions = to_device(t[:action])
action_log_probs = t[:action_log_prob]

if p.jointPPO
    action_log_probs = sum(action_log_probs, dims=1)[:]
else
    action_log_probs = reshape(action_log_probs, 1, size(action_log_probs)[1], size(action_log_probs)[2])
end

stop_update = false

if p.one_by_one_training
    n_epochs = n_epochs * p.n_actors
end

rand_actor_inds = shuffle!(rng, Vector(1:p.n_actors))
reverse_actor_inds = Vector(p.n_actors:-1:1)

actor_losses = Float32[]
critic_losses = Float32[]







epoch = 1

rand_inds = shuffle!(rng, Vector(1:n_rollout))

i = 1

inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

#global s, a, r, log_p, adv

s = to_device(collect(select_last_dim(states, inds)))
a = to_device(collect(select_last_dim(actions, inds)))

r = select_last_dim(returns, inds)
log_p = select_last_dim(action_log_probs, inds)
adv = select_last_dim(advantages, inds)

clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-5 to avoid inf

if p.normalize_advantage
    adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
end



#global obs_rep, v′, temp_act, μ, logσ, log_p′ₐ, ratio

obs_rep, v′ = p.encoder(s)

# obs_rep, v′_no = p.encoder(s)
# obs_rep_no, v′ = encoder(s)

#parallel act
# temp_act = cat(zeros(Float32,1,1,size(a)[3]),a[:,1:end-1,:],dims=2)
# μ, logσ = p.decoder(temp_act, obs_rep)

#   auto regressive act

μ, logσ = p.decoder(zeros(Float32,na,1,microbatch_size), obs_rep[:,1:1,:])

for n in 2:p.n_actors
    newμ, newlogσ = p.decoder(cat(zeros(Float32,na,1,microbatch_size), μ, dims=2), obs_rep[:,1:n,:])

    μ = cat(μ, newμ[:,end:end,:], dims=2)
end

log_p′ₐ = sum(normlogpdf(μ, exp.(logσ), a), dims=1)


if p.jointPPO
    log_p′ₐ = sum(log_p′ₐ, dims=2)[:]
end

ratio = exp.(log_p′ₐ .- log_p)


approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

if approx_kl_div > p.target_kl && (i > 1 || epoch > 1) # only in second batch
    println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
    stop_update = true
end

#adv = reshape(adv, 1, size(adv)[1], size(adv)[2])
#r = reshape(r, 1, size(r)[1], size(r)[2])




if p.jointPPO
    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

    surr1 = ratio .* adv
    surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

    actor_loss = -mean(min.(surr1, surr2))

    critic_loss = mean((r .- v′[1,1,:]) .^ 2)

elseif p.one_by_one_training
    temp_index = 1 + epoch%p.n_actors
    #actor_index = rand_actor_inds[temp_index]
    #actor_index = reverse_actor_inds[temp_index]
    actor_index = temp_index

    entropy_loss = mean(size(logσ[:,actor_index,:], 1) * (log(2.0f0π) + 1) .+ sum(logσ[:,actor_index,:]; dims=1)) / 2

    surr1 = ratio[1,actor_index,:] .* adv[actor_index,:]
    surr2 = clamp.(ratio[1,actor_index,:], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv[actor_index,:]

    actor_loss = -mean(min.(surr1, surr2))

    critic_loss = mean((r[actor_index,:] .- v′[1,actor_index,:]) .^ 2)

else
    entropy_loss = mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2

    surr1 = ratio[1,:,:] .* adv
    surr2 = clamp.(ratio[1,:,:], 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

    actor_loss = -mean(min.(surr1, surr2))

    critic_loss = mean((r .- v′[1,:,:]) .^ 2)
end



loss = w₁ * actor_loss + w₂ * critic_loss - w₃ * entropy_loss

ignore() do
    push!(actor_losses, w₁ * actor_loss)
    push!(critic_losses, w₂ * critic_loss)
end

loss
        


mean_actor_loss = mean(abs.(actor_losses))
mean_critic_loss = mean(abs.(critic_losses))
println("---")
println(mean_actor_loss)
println(mean_critic_loss)


if p.adaptive_weights
    actor_factor = clamp(0.5/mean_actor_loss, 0.9, 1.1)
    critic_factor = clamp(0.3/mean_critic_loss, 0.9, 1.1)
    println("changing actor weight from $(w₁) to $(w₁*actor_factor)")
    println("changing critic weight from $(w₂) to $(w₂*critic_factor)")
    p.actor_loss_weight = w₁ * actor_factor
    p.critic_loss_weight = w₂ * critic_factor
end

println("---")