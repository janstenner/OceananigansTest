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
    
end