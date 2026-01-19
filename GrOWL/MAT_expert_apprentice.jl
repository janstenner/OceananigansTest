using Zygote
using Optimisers
using Flux


# load()
# load(701) # for window size 7
# agent.policy.approximator.actor.logσ[1] = -14.0f0

batch_size = 20

growl_power = 0.00016
growl_freq = 1
growl_srate = 0.9

group_rows_by_overlap = true
group_channels = true

training_steps = 35_000
extra_steps = 0

block_num = 1
dim_model = 22
head_num = 2
head_dim = 11
ffn_dim = 22
drop_out = 0.00#1

betas = (0.9, 0.999)

customCrossAttention = true
jointPPO = false
one_by_one_training = false
positional_encoding = 3 #ZeroEncoding

square_rewards = false
randomIC = false

apprentice_agent = create_agent_mat(n_actors = actuators,
                    action_space = actionspace,
                    state_space = env.state_space,
                    use_gpu = false, 
                    rng = rng,
                    y = y, p = p,
                    start_steps = start_steps, 
                    start_policy = start_policy,
                    update_freq = update_freq,
                    learning_rate = 3e-4,
                    nna_scale = 1.0,
                    nna_scale_critic = 1.0,
                    drop_middle_layer = true,
                    drop_middle_layer_critic = true,
                    fun = gelu,
                    clip1 = false,
                    n_epochs = n_epochs,
                    n_microbatches = n_microbatches,
                    logσ_is_network = false,
                    max_σ = max_σ,
                    entropy_loss_weight = entropy_loss_weight,
                    adaptive_weights = false,
                    clip_grad = clip_grad,
                    target_kl = target_kl,
                    start_logσ = -10.0f0,
                    dim_model = dim_model,
                    block_num = block_num,
                    head_num = head_num,
                    head_dim = head_dim,
                    ffn_dim = ffn_dim,
                    drop_out = drop_out,
                    betas = betas,
                    jointPPO = jointPPO,
                    customCrossAttention = customCrossAttention,
                    one_by_one_training = one_by_one_training,
                    clip_range = clip_range,
                    tanh_end = tanh_end,
                    positional_encoding = positional_encoding,
                    )


apprentice = apprentice_agent.policy

encoder = apprentice.encoder
decoder = apprentice.decoder

mask = ones(Float32, size(env.state[:,1]))




function update_mask(threshold = 0.1)
    global mask

    mask = ones(Float32, size(env.state[:,1]))

    # first_layer_matrix = apprentice.encoder.embedding.weight

    # back_projection = zeros(size(env.state[:,1]))

    # for i in 1:size(first_layer_matrix)[1]
    #     for j in 1:size(first_layer_matrix)[2]
    #         back_projection[j] += abs(first_layer_matrix[i,j])
    #     end
    # end

    # another, more simple way
    transposed_weights = transpose(apprentice.encoder.embedding.weight)
    transposed_weights = abs.(transposed_weights)
    back_projection = sum(transposed_weights, dims=2)[:]

    indexes_to_be_zero = findall(x -> x <= threshold, back_projection)

    println("Number of indexes to be zero: ", length(indexes_to_be_zero))

    mask[indexes_to_be_zero] .= 0.0f0
end


function plot_masked_input()
    global mask

    first_layer_matrix = apprentice.encoder.embedding.weight

    back_projection = zeros(size(env.state[:,1]))

    for i in 1:size(first_layer_matrix)[1]
        for j in 1:size(first_layer_matrix)[2]
            back_projection[j] += abs(first_layer_matrix[i,j])
        end
    end

    temp_y = reshape(back_projection .* mask, 3,window_size,sensors[2]+1)

    p = make_subplots(rows=1, cols=3)

    add_trace!(p, heatmap(z=temp_y[1,:,:]', coloraxis="coloraxis"), col = 1)
    add_trace!(p, heatmap(z=temp_y[2,:,:]', coloraxis="coloraxis"), col = 2)
    add_trace!(p, heatmap(z=temp_y[3,:,:]', coloraxis="coloraxis"), col = 3)

    colorscale = [[0, "rgb(0, 0, 0)"], [0.01, "rgb(140, 90, 230)"], [1, "rgb(190, 120, 255)"], ]

    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 0, cmax = maximum(temp_y), colorscale = colorscale),
        )


    relayout!(p, layout.fields)

    display(p)

    # now plot the overlayed windows of all agents and the sensor counts by channels

    sensor_window = reshape(mask, 3, window_size, sensors[2]+1)
    total_sensors = zeros(3, sensors[1], sensors[2]+1)
    window_half_size = Int(floor(window_size/2))

    for i in actuators_to_sensors
        temp_indexes = [(i + j + sensors[1] - 1) % sensors[1] + 1 for j in 0-window_half_size:0+window_half_size]

        total_sensors[:, temp_indexes, :] .+= sensor_window
    end

    total_sensors = clamp.(total_sensors, 0.0f0, 1.0f0)
    total_sensors_combined = total_sensors[1,:,:] + total_sensors[2,:,:] + total_sensors[3,:,:]

    p = plot(heatmap(z=total_sensors_combined'))
    display(p)


    indexes_zero = findall(x -> x == 0.0, mask)
    println("Sparsity: $(100*length(indexes_zero)/length(mask))%")

    combined = total_sensors_combined[:]
    indexes_zero_combined = findall(x -> x == 0.0, combined)
    println("Sparsity combined channels: $(length(indexes_zero_combined)/length(combined))%")
end


function generate_states()
    global states

    states = zeros(Float32, size(env.state)[1], size(env.state)[2], 100)

    reset!(env)
    generate_random_init()

    for i in 1:100

        #action = agent(env)
        action = prob(agent.policy, env.state, nothing).μ

        states[:, :, i] .= env.state

        env(action)

        # println(i, "% of simulation done")
    end
end

function growl_train(;training_steps = training_steps, extra_steps = extra_steps, growl=true, group_rows_by_overlap = group_rows_by_overlap, group_channels = group_channels)

    global states

    if !(@isdefined states)
        generate_states()
    end

    global row_groups

    row_groups = get_row_groups(;group_channels = group_channels)


    global losses = Float32[]
    for i in 1:training_steps+extra_steps
        
        i%100 == 0 && i <= training_steps && println(i*100/training_steps, "% done")

        i == training_steps+1 && println("training_steps finished, starting extra_steps...")
        i%100 == 0 && i > training_steps && println((i-training_steps)*100/extra_steps, "% of extra steps done")

        # training call
        rand_inds = shuffle!(rng, Vector(1:100))
        for j in 1:Int(100/batch_size)
            #println("j is $(j) of $(Int(100/batch_size))")
            global batch = states[:, :, rand_inds[(j-1)*batch_size+1:j*batch_size]]
            batch_masked = batch .* mask

            na = size(apprentice.decoder.embedding.weight)[2]

            global g_encoder
            global g_decoder

            g_encoder, g_decoder = Flux.gradient(apprentice.encoder, apprentice.decoder) do p_encoder, p_decoder

                obsrep, val = p_encoder(batch_masked)

                # μ, logσ = p_decoder(zeros(Float32,na,1,batch_size), obsrep[:,1:1,:])

                # for n in 2:apprentice.n_actors
                #     newμ, newlogσ = p_decoder(cat(zeros(Float32,na,1,batch_size), μ, dims=2), obsrep[:,1:n,:])

                #     μ = cat(μ, newμ[:,end:end,:], dims=2)
                # end

                # diff = μ - agent.policy.approximator.actor(batch)[1]


                # new variant
                μ_expert = prob(agent.policy, batch, nothing).μ

                temp_act = cat(zeros(Float32,na,1,batch_size),μ_expert[:,1:end-1,:],dims=2)
                μ, logσ = p_decoder(temp_act, obsrep)

                diff = μ - μ_expert
                mse = mean(diff.^2)

                # Zygote.@ignore println(mse)

                mse
            end

            Flux.update!(apprentice.encoder_state_tree, apprentice.encoder, g_encoder)
            Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder)



            if i%growl_freq == 0 && growl && i <= training_steps
                # GroWL routine

                # println("starting GrOWL training...")

                # weights_before = deepcopy(apprentice.encoder.embedding.weight)
                apply_growl(apprentice.encoder.embedding.weight;  group_rows_by_overlap = group_rows_by_overlap)
                # difference = sum(abs.(weights_before - apprentice.encoder.embedding.weight))

                # println(difference)


                #keep the zeros if this is the last growl step
                if i+growl_freq > training_steps
                    println("keeping the zeros in the last grOWL step")
                    update_mask(0.0)
                end
            end
        end


        if i%100 == 0 
            transposed_weights = transpose(apprentice.encoder.embedding.weight)
            n_rows = size(transposed_weights, 1)
            zero_row_idcs = [i for i in 1:n_rows if all(transposed_weights[i, :] .== 0)]
            
            println("zero inputs: $(length(zero_row_idcs))")
            weight_factor = sum(abs.(apprentice.encoder.embedding.weight))
            println("weight factor: $(weight_factor)")
        end


        #check current performance of the apprentice
        diff = prob(apprentice, states .* mask, nothing).μ - prob(agent.policy, states, nothing).μ
        mse = sum(diff.^2)
        push!(losses, mse)
    end

    plot(losses)
end






function get_row_groups(;group_channels = true)

    row_groups = []

    index_array = collect(1:size(env.state[:,1])[1])

    index_y = reshape(index_array, 3,window_size,sensors[2]+1)

    # create stencil for grouping
    center_point = Int(ceil(window_size/2))
    agent_delta = Int(sensors[1] / actuators)


    anchor_steps = Int(ceil(center_point/agent_delta)+1)
    
    if group_channels
        stencil_index_array = collect(1:agent_delta*(sensors[2]+1))
        index_stencil = reshape(stencil_index_array, 1, agent_delta, sensors[2]+1)

        anchors = [
            [1,2,3],
            [center_point + (j * agent_delta) for j in -anchor_steps:anchor_steps],
            [1]
        ]
    else
        stencil_index_array = collect(1:3*agent_delta*(sensors[2]+1))
        index_stencil = reshape(stencil_index_array, 3, agent_delta, sensors[2]+1)

        anchors = [
            [1],
            [center_point + (j * agent_delta) for j in -anchor_steps:anchor_steps],
            [1]
        ]
    end
    
    for i in stencil_index_array
        # get the stencil offset for the current index
        stencil_offset = collect(findfirst(x -> x == i, index_stencil).I .- 1)

        # get the anchor points for the current stencil offset
        anchor_points = deepcopy(anchors)
        anchor_points[1] .+=  stencil_offset[1]
        anchor_points[2] .+=  stencil_offset[2]
        anchor_points[3] .+=  stencil_offset[3]

        # filter for valid indices
        anchor_points[1] = filter(i -> (1 ≤ i ≤ size(index_y, 1)), anchor_points[1])
        anchor_points[2] = filter(i -> (1 ≤ i ≤ size(index_y, 2)), anchor_points[2])
        anchor_points[3] = filter(i -> (1 ≤ i ≤ size(index_y, 3)), anchor_points[3])

        # get the indices of the row group
        push!(row_groups, index_y[anchor_points...][:])
    end

    return row_groups
end

row_groups = get_row_groups(group_channels = group_channels)


# utility function to check for duplicates of row_groups. Should return false
function any_shared(c)
    seen = Set{Int}()

    for arr in c
        for x in arr
            if x in seen
                return true              # x appeared in a previous sub‐array
            end
            push!(seen, x)
        end
    end
    return false                        # no element was seen twice
end




function apply_growl(model_weights; group_rows_by_overlap = true)

    pl_srate = growl_srate

    reshaped_weight = transpose(model_weights)

    global row_groups

    if group_rows_by_overlap
        groups = deepcopy(row_groups)
    else
        groups =  [[i] for i in 1:size(reshaped_weight, 1)]
    end

    # Compute the L2 norm for each row.
    n_groups = length(groups)
    n2_groups = [norm(reshaped_weight[i, :][:], 2) for i in groups]

    # --- Create GrOWL parameters ---
    # Sort the row norms (returns indices that sort in increasing order).
    s_inds = sortperm(n2_groups)
    # Generate theta parameters (user-supplied function).
    theta_is = ones(n_groups) * 0.2
    theta_is[1:Int(floor(0.6 * n_groups))] .= 1.0
    theta_is = ones(n_groups)
    # make the parameters smaller in general
    theta_is .*= growl_power

    # Apply the proximal operator.
    new_n2_groups = proxOWL(deepcopy(n2_groups), deepcopy(theta_is))

    # --- Rescale the weight rows ---
    new_W = similar(reshaped_weight)
    eps_val = eps(Float32)

    for i in 1:n_groups

        if new_n2_groups[i] < eps_val
            # If the norm is too small, set all rows belonging to the group to zero.
            for j in groups[i]
                new_W[j, :] .= zeros(eltype(reshaped_weight), size(reshaped_weight, 2))
            end
        else
            # Scale all rows belonging to the group.
            for j in groups[i]
                new_W[j, :] .= reshaped_weight[j, :] .* (new_n2_groups[i] / n2_groups[i])
            end       
        end
    end


    # --- Check for excessive pruning ---

    # Find indices of groups that are entirely zero.
    zero_group_idcs = [i for i in 1:n_groups if new_n2_groups[i] < eps_val]

    max_slct = Int(floor(pl_srate * n_groups))

    if length(zero_group_idcs) > max_slct

        numel = length(zero_group_idcs)
        shuffled_idcs = shuffle(1:numel)
        use_slct = numel - max_slct
        selected_idcs = shuffled_idcs[1:use_slct]
        selected_elmts = zero_group_idcs[selected_idcs]


        for i in selected_elmts
            # Restore all rows belonging to the group.
            for j in groups[i]
                new_W[j, :] .= reshaped_weight[j, :]
            end
        end
    end

    new_W = transpose(new_W)

    # Update the weight in the model (assumes in-place update is acceptable).
    model_weights .= new_W
end



function proxOWL(z::Vector, mu::Vector)
    # store the signs of z.
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
        v1 = deepcopy(z_sorted[1:k])
        v2 = deepcopy(mu[1:k])
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

        action = prob(apprentice, env.state .* mask, nothing).μ[:,:,1]

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

    p = plot(rewards)
    display(p)



    if true
        isdir("video_output") || mkdir("video_output")
        rm("video_output/MAT_Apprentice.mp4", force=true)
        #run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "video_output/MAT_Apprentice.mp4"`)

        run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "video_output/MAT_Apprentice.mp4"`)
    end
end


#growl_train(training_steps)

#render_run_apprentice()




#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "saves/*")
end

function load(number = nothing)
    if isnothing(number)
        global apprentice = FileIO.load(dirpath * "/saves/MAT_Apprentice.jld2","apprentice")
    else
        global apprentice = FileIO.load(dirpath * "/saves/MAT_Apprentice$number.jld2","apprentice")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/MAT_Apprentice.jld2","apprentice",apprentice)
    else
        FileIO.save(dirpath * "/saves/MAT_Apprentice$number.jld2","apprentice",apprentice)
    end
end





function train_masked(use_random_init = true; visuals = false, num_steps = 1600, inner_loops = 5, outer_loops = 1)
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

                while !(is_terminated(env) || is_truncated(env))

                    # update env state!!!!!
                    env.state = env.state .* mask

                    action = agent(env)

                    agent(PRE_ACT_STAGE, env, action)
                    hook(PRE_ACT_STAGE, agent, env, action)

                    env(action)

                    agent(POST_ACT_STAGE, env)
                    hook(POST_ACT_STAGE, agent, env)

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

                if is_terminated(env) || is_truncated(env)
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


function load_masked(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/masked_hookMAT.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/masked_agentMAT.jld2","agent")
        global mask = FileIO.load(dirpath * "/saves/masked_maskMAT.jld2","mask")
    else
        global hook = FileIO.load(dirpath * "/saves/masked_hookMAT$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/masked_agentMAT$number.jld2","agent")
        global mask = FileIO.load(dirpath * "/saves/masked_maskMAT$number.jld2","mask")
    end
end

function save_masked(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/masked_hookMAT.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/masked_agentMAT.jld2","agent",agent)
        FileIO.save(dirpath * "/saves/masked_maskMAT.jld2","mask",mask)
    else
        FileIO.save(dirpath * "/saves/masked_hookMAT$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/masked_agentMAT$number.jld2","agent",agent)
        FileIO.save(dirpath * "/saves/masked_maskMAT$number.jld2","mask",mask)
    end
end