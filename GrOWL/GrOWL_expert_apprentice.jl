using Zygote
using Optimisers
using Flux


load()
agent.policy.approximator.actor.logσ[1] = -14.0f0

batch_size = 20

growl_power = 0.005
growl_freq = 70
growl_srate = 0.9

total_steps = 4_000

apprentice = Chain(Dense(size(env.state)[1], 64, gelu), Dense(64, 64, gelu), Dense(64, 1))

new_learning_rate = 1e-4
optimizer_apprentice = Optimisers.OptimiserChain(Optimisers.ClipNorm(clip_grad), Optimisers.Adam(new_learning_rate, betas))
apprentice_state_tree = Flux.setup(optimizer_apprentice, apprentice)




function growl_train(total_steps = 1_000; growl=true)

    global states

    if isnothing(states)
        states = zeros(Float32, size(env.state)[1], size(env.state)[2], 100)

        reset!(env)
        for i in 1:100

            action = agent(env)

            states[:, :, i] .= env.state

            env(action)

            println(i, "% of simulation done")
        end
    end


    global losses = Float32[]
    for i in 1:total_steps
        
        i%100 == 0 && println(i*100/total_steps, "% done")

        # training call
        rand_inds = shuffle!(rng, Vector(1:100))
        for j in 1:Int(100/batch_size)
            batch = states[:, :, rand_inds[(j-1)*batch_size+1:j*batch_size]]

            g_apprentice = Flux.gradient(apprentice) do appr
                diff = appr(batch) - agent.policy.approximator.actor(batch)[1]
                mse = mean(diff.^2)
                return mse
            end

            Flux.update!(apprentice_state_tree, apprentice, g_apprentice[1])
        end

        if i%growl_freq == 0 && growl
            # GroWL routine
            println("starting GrOWL training...")
            weights_before = deepcopy(apprentice.layers[1].weight)
            apply_growl(apprentice.layers[1].weight)
            difference = sum(abs.(weights_before - apprentice.layers[1].weight))
            println(difference)
            transposed_weights = transpose(apprentice.layers[1].weight)
            n_rows = size(transposed_weights, 1)
            zero_row_idcs = [i for i in 1:n_rows if all(transposed_weights[i, :] .== 0)]
            println(length(zero_row_idcs))
        end


        #check current performance of the apprentice
        diff = apprentice(states) - agent.policy.approximator.actor(states)[1]
        mse = sum(diff.^2)
        push!(losses, mse)
    end

    plot(losses)
end





function apply_growl(model_weights)

    pl_srate = growl_srate

    reshaped_weight = transpose(model_weights)

    # Compute the L2 norm for each row.
    n_rows = size(reshaped_weight, 1)
    n2_rows_W = [norm(reshaped_weight[i, :], 2) for i in 1:n_rows]

    # --- Create GrOWL parameters ---
    # Sort the row norms (returns indices that sort in increasing order).
    s_inds = sortperm(n2_rows_W)
    # Generate theta parameters (user-supplied function).
    theta_is = ones(n_rows) * 0.2
    theta_is[1:Int(floor(0.6 * n_rows))] .= 1.0
    theta_is = ones(n_rows)
    # make the parameters smaller in general
    theta_is .*= growl_power

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



function proxOWL(z::Vector, mu::Vector)
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



function proxOWL_segments(A::Vector, B::Vector)
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


function render_run_apprentice()
    global rewards = Float64[]
    global collected_actions = zeros(200,actuators)
    reward_sum = 0.0

    rm("frames/", recursive=true, force=true)
    mkdir("frames")

    colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
    ymax = 30
    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
        )


    reset!(env)
    generate_random_init()

    for i in 1:200

        action = apprentice(env.state)

        collected_actions[i,:] = action[:]
        env(action)

        result = env.y[1,:,:]
        result_W = env.y[2,:,:]
        result_U = env.y[3,:,:]

        p = make_subplots(rows=1, cols=1)

        add_trace!(p, heatmap(z=result', coloraxis="coloraxis"), col = 1)
        #add_trace!(p, heatmap(z=result_W'), col = 2)
        #add_trace!(p, heatmap(z=result_U'), col = 3)

        # p = plot(heatmap(z=result', coloraxis="coloraxis"), layout)

        relayout!(p, layout.fields)

        savefig(p, "frames/a$(lpad(string(i), 4, '0')).png"; width=1600, height=800)
        #body!(w,p)

        temp_reward = reward_function(env; returnGlobalNu = true)
        println(temp_reward)

        reward_sum += temp_reward
        push!(rewards, temp_reward)

        # println(mean(env.reward))

        # reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))
    end

    println(reward_sum)



    if true
        isdir("video_output") || mkdir("video_output")
        rm("video_output/$scriptname.mp4", force=true)
        #run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "video_output/$scriptname.mp4"`)

        run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "video_output/$scriptname.mp4"`)
    end
end


growl_train(total_steps)

#render_run_apprentice()