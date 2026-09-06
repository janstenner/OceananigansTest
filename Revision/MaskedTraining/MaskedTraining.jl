using RL

"""Expose masked observations only to the agent; restore full state even on failure."""
function with_masked_state(f, env, mask)
    full_state = env.state
    env.state = full_state .* mask
    try
        return f()
    finally
        env.state = full_state
    end
end

function train_masked!(agent, hook, env, mask, episodes; progress_every = 100)
    episodes >= 0 || error("Negative episode budget")
    episodes == 0 && return 0
    steps = 0
    hook(PRE_EXPERIMENT_STAGE, agent, env)
    with_masked_state(() -> agent(PRE_EXPERIMENT_STAGE, env), env, mask)
    for episode in 1:episodes
        reset!(env)
        with_masked_state(() -> agent(PRE_EPISODE_STAGE, env), env, mask)
        hook(PRE_EPISODE_STAGE, agent, env)
        while !(is_terminated(env) || is_truncated(env))
            action = with_masked_state(env, mask) do
                a = agent(env)
                agent(PRE_ACT_STAGE, env, a)
                a
            end
            hook(PRE_ACT_STAGE, agent, env, action)
            # GeneralEnv computes reward inside this call. Full state and all
            # physical sensor fields are available throughout the PDE step.
            env(action)
            with_masked_state(() -> agent(POST_ACT_STAGE, env), env, mask)
            hook(POST_ACT_STAGE, agent, env)
            steps += 1
        end
        with_masked_state(() -> agent(POST_EPISODE_STAGE, env), env, mask)
        hook(POST_EPISODE_STAGE, agent, env)
        if progress_every > 0 && (episode % progress_every == 0 || episode == episodes)
            println("Episode $episode/$episodes, reward $(hook.rewards[end])")
            flush(stdout)
        end
    end
    hook(POST_EXPERIMENT_STAGE, agent, env)
    length(hook.rewards) == episodes || error("Episode count mismatch")
    steps
end
