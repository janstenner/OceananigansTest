using Zygote





function growl_train(use_random_init = true; visuals = false, num_steps = 1600, inner_loops = 5, outer_loops = 1)
    rm(dirpath * "/training_frames/", recursive=true, force=true)
    mkdir(dirpath * "/training_frames/")
    frame = 1

    if visuals
        colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
        ymax = 30
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
            )
    end


    if use_random_init
        hook.generate_random_init = generate_random_init
    else
        hook.generate_random_init = false
    end
    

    for i = 1:outer_loops
        
        for i = 1:inner_loops
            println("")
            
            stop_condition = StopAfterEpisodeWithMinSteps(num_steps)


            # run start
            hook(PRE_EXPERIMENT_STAGE, agent, env)
            agent(PRE_EXPERIMENT_STAGE, env)
            is_stop = false
            while !is_stop
                reset!(env)
                agent(PRE_EPISODE_STAGE, env)
                hook(PRE_EPISODE_STAGE, agent, env)

                while !is_terminated(env) # one episode
                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

                    # hack to prevent normal training
                    agent.policy.update_step = 1

                    # growl training call
                    if frame%200 == 0
                        growl_update!(agent.policy, agent.trajectory)
                    end

                    if visuals
                        p = plot(heatmap(z=env.y[1,:,:]', coloraxis="coloraxis"), layout)

                        savefig(p, dirpath * "/training_frames//a$(lpad(string(frame), 5, '0')).png"; width=1000, height=800)
                    end

                    frame += 1

                    if stop_condition(agent, env)
                        is_stop = true
                        break
                    end
                end # end of an episode

                if is_terminated(env)
                    agent(POST_EPISODE_STAGE, env)  # let the agent see the last observation
                    hook(POST_EPISODE_STAGE, agent, env)
                end
            end
            hook(POST_EXPERIMENT_STAGE, agent, env)
            # run end


            println(hook.bestreward)
            

            # hook.rewards = clamp.(hook.rewards, -3000, 0)
        end
    end

    if visuals && false
        rm(dirpath * "/training.mp4", force=true)
        run(`ffmpeg -framerate 16 -i $(dirpath * "/training_frames/a%05d.png") -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le $(dirpath * "/training.mp4")`)
    end

    #save()
end



function growl_update!(p::PPOPolicy, t::Any)
    rng = p.rng
    AC = p.approximator
    γ = p.γ
    λ = p.λ
    n_epochs = p.n_epochs
    n_microbatches = p.n_microbatches
    clip_range = p.clip_range
    w₁ = p.actor_loss_weight
    w₂ = p.critic_loss_weight
    w₃ = p.entropy_loss_weight
    D = RL.device(AC)
    to_device(x) = send_to_device(D, x)

    n_envs, n_rollout = size(t[:terminal])
    @assert n_envs * n_rollout % n_microbatches == 0 "size mismatch"
    microbatch_size = n_envs * n_rollout ÷ n_microbatches

    n = length(t)
    states = to_device(t[:state])


    states_flatten_on_host = flatten_batch(select_last_dim(t[:state], 1:n))

    values = reshape(send_to_host(AC.critic(flatten_batch(states))), n_envs, :)
    next_values = reshape(flatten_batch(t[:next_values]), n_envs, :)

    advantages = generalized_advantage_estimation(
        t[:reward],
        values,
        next_values,
        γ,
        λ;
        dims=2,
        terminal=t[:terminal]
    )
    returns = to_device(advantages .+ select_last_dim(values, 1:n_rollout))
    advantages = to_device(advantages)

    actions_flatten = flatten_batch(select_last_dim(t[:action], 1:n))
    action_log_probs = select_last_dim(to_device(t[:action_log_prob]), 1:n)

    stop_update = false

    for epoch in 1:n_epochs

        rand_inds = shuffle!(rng, Vector(1:n_envs*n_rollout))
        for i in 1:n_microbatches

            inds = rand_inds[(i-1)*microbatch_size+1:i*microbatch_size]

            # s = to_device(select_last_dim(states_flatten_on_host, inds))
            # !!! we need to convert it into a continuous CuArray otherwise CUDA.jl will complain scalar indexing
            s = to_device(collect(select_last_dim(states_flatten_on_host, inds)))
            a = to_device(collect(select_last_dim(actions_flatten, inds)))

            if eltype(a) === Int
                a = CartesianIndex.(a, 1:length(a))
            end

            r = vec(returns)[inds]
            log_p = vec(action_log_probs)[inds]
            adv = vec(advantages)[inds]

            clamp!(log_p, log(1e-8), Inf) # clamp old_prob to 1e-5 to avoid inf

            if p.normalize_advantage
                adv = (adv .- mean(adv)) ./ clamp(std(adv), 1e-8, 1000.0)
            end

            if isnothing(AC.actor_state_tree)
                AC.actor_state_tree = Flux.setup(AC.optimizer_actor, AC.actor)
            end

            if isnothing(AC.critic_state_tree)
                AC.critic_state_tree = Flux.setup(AC.optimizer_critic, AC.critic)
            end

            g_actor = Flux.gradient(AC.actor) do actor
                v′ = AC.critic(s) |> vec
                if actor isa GaussianNetwork
                    μ, logσ = actor(s)
                    
                    if ndims(a) == 2
                        log_p′ₐ = vec(sum(normlogpdf(μ, exp.(logσ), a), dims=1))
                    else
                        log_p′ₐ = normlogpdf(μ, exp.(logσ), a)
                    end
                    entropy_loss =
                        mean(size(logσ, 1) * (log(2.0f0π) + 1) .+ sum(logσ; dims=1)) / 2
                else
                    # actor is assumed to return discrete logits
                    logit′ = actor(s)

                    p′ = softmax(logit′)
                    log_p′ = logsoftmax(logit′)
                    log_p′ₐ = log_p′[a]
                    entropy_loss = -sum(p′ .* log_p′) * 1 // size(p′, 2)
                end
                ratio = exp.(log_p′ₐ .- log_p)

                ignore() do
                    approx_kl_div = mean((ratio .- 1) - log.(ratio)) |> send_to_host

                    if approx_kl_div > p.target_kl
                        println("Target KL overstepped: $(approx_kl_div) at epoch $(epoch), batch $(i)")
                        stop_update = true
                    end
                end

                surr1 = ratio .* adv
                surr2 = clamp.(ratio, 1.0f0 - clip_range, 1.0f0 + clip_range) .* adv

                actor_loss = -mean(min.(surr1, surr2))
                #critic_loss = mean((r .- v′) .^ 2)
                loss = w₁ * actor_loss #- w₃ * entropy_loss   # also exclude entropy loss for now

                # println("-------------")
                # println(w₁ * actor_loss)
                # println(w₂ * critic_loss)
                # println(w₃ * entropy_loss)

                loss
            end
            
            if !stop_update
                Flux.update!(AC.actor_state_tree, AC.actor, g_actor)
                #Flux.update!(AC.critic_state_tree, AC.critic, g_critic)
            else
                break
            end

        end

        if stop_update
            break
        end
    end


    # GroWL routine
    apply_growl(agent.policy.approximator.actor.μ.layers[1].weight)
end






function apply_growl(model_weights)

    pl_srate = 0.9

    reshaped_weight = transpose(model_weights)

    # Compute the L2 norm for each row.
    n_rows = size(reshaped_weight, 1)
    n2_rows_W = [norm(reshaped_weight[i, :], 2) for i in 1:n_rows]

    # --- Create GrOWL parameters ---
    # Sort the row norms (returns indices that sort in increasing order).
    s_inds = sortperm(n2_rows_W)
    # Generate theta parameters (user-supplied function).
    theta_is = ones(n_rows) * 0.2
    theta_is[1] = 1.0

    # Apply the proximal operator.
    new_n2_rows_W = proxOWL(copy(n2_rows_W), copy(theta_is))

    # --- Rescale the weight rows ---
    new_W = similar(reshaped_weight)
    eps_val = eps(Float32)
    for i in 1:n_rows
        if n2_rows_W[i] < eps_val
            new_W[i, :] .= zeros(eltype(reshaped_weight), size(reshaped_weight, 2))
        else
            new_W[i, :] .= reshaped_weight[i, :] .* (new_n2_rows_W[i] / n2_rows_W[i])
        end
    end

    # --- Check for excessive pruning ---
    # Find indices of rows that are entirely zero.
    zero_row_idcs = [i for i in 1:n_rows if all(new_W[i, :] .== 0)]
    max_slct = Int(floor(pl_srate * n_rows))
    if length(zero_row_idcs) > max_slct
        numel = length(zero_row_idcs)
        shuffled_idcs = shuffle(1:numel)
        use_slct = numel - max_slct
        selected_idcs = shuffled_idcs[1:use_slct]
        selected_elmts = zero_row_idcs[selected_idcs]
        for i in selected_elmts
            new_W[i, :] .= reshaped_weight[i, :]
        end
    end

    new_W = transpose(new_W)

    # Update the weight in the model (assumes in-place update is acceptable).
    model_weights .= new_W
end



function proxOWL(z::Vector{Float64}, mu::Vector{Float64})
    # Restore the signs of z.
    sgn = sign.(z)
    # Work with absolute values.
    z_abs = abs.(z)
    # Sort z_abs in non-increasing (descending) order.
    indx = sortperm(z_abs, rev=true)
    z_sorted = z_abs[indx]
    n = length(z_sorted)
    x = zeros(n)
    diff = z_sorted .- mu
    # Reverse diff to mimic Python’s diff[::-1]
    diff_rev = reverse(diff)
    # Find the first index in the reversed diff that is > 0.
    indc = findfirst(x -> x > 0, diff_rev)
    flag = indc === nothing ? 0.0 : diff_rev[indc]
    if flag > 0
        # In Python: k = n - indc, but note the 1-index adjustment in Julia.
        k = n - indc + 1
        v1 = copy(z_sorted[1:k])
        v2 = copy(mu[1:k])
        v = proxOWL_segments(v1, v2)
        # Prepare an output array in original order.
        x_orig = zeros(n)
        for j in 1:k
            # indx[j] holds the original index for the j-th largest element.
            x_orig[indx[j]] = v[j]
        end
        x = x_orig
    end
    # Restore original signs.
    x = sgn .* x
    return x
end



function proxOWL_segments(A::Vector{Float64}, B::Vector{Float64})
    modified = true
    k = 0
    max_its = 1000
    # Loop until no modifications occur or we exceed the maximum iterations.
    while modified && k <= max_its
        modified = false
        segments = Tuple{Int,Int}[]
        new_start = true
        start_idx = nothing
        end_idx = nothing

        for i in 1:length(A)-1
            if (A[i] - B[i] > 0) && (A[i+1] - B[i+1] > 0)
                if (A[i] - B[i] < A[i+1] - B[i+1])
                    modified = true
                    if new_start
                        start_idx = i
                        new_start = false
                    end
                    continue
                elseif (A[i] - B[i] >= A[i+1] - B[i+1])
                    if start_idx !== nothing
                        end_idx = i
                        push!(segments, (start_idx, end_idx))
                    end
                    new_start = true
                    start_idx = nothing
                    end_idx = nothing
                end
            end
        end

        # If a segment was started but not ended, finish it.
        if (start_idx !== nothing) && (end_idx === nothing)
            end_idx = length(A)
            push!(segments, (start_idx, end_idx))
        end

        # If no segments were found, exit the loop.
        if isempty(segments)
            break
        end

        # For each segment, replace A and B over that range with their means.
        for (s, e) in segments
            avg_A = mean(A[s:e])
            avg_B = mean(B[s:e])
            for j in s:e
                A[j] = avg_A
                B[j] = avg_B
            end
            modified = true
        end
        k += 1
    end

    # Compute X = A - B and set any negative values to zero.
    X = A .- B
    X = map(x -> x < 0 ? 0.0 : x, X)
    return X
end