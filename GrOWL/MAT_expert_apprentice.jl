using Zygote
using Optimisers
using Flux




include("../rIC-validation.jl")
include("../fixedIC-validation.jl")




# rIC scores
#reward_sums = [554.6305286939099, 560.2386884968918, 565.4618377650218, 557.434398548902, 561.6262672069304, 557.466004557712, 555.4317062304078, 555.4464920992392, 572.1192004805054, 548.639008840305, 548.6114441745203, 550.9900855238976, 556.3108301071123, 585.3668568567299, 552.5313324045303]
#reward_sums_apprentice_growl = [553.6471336620853, 559.751815049584, 566.2764691470024, 556.4568649930598, 563.2828020581873, 556.4910212653255, 555.1814424620101, 555.1843951026261, 575.2583532783839, 549.4435999495879, 549.4232846558572, 552.0851559461055, 555.9527187866161, 566.501420245082, 552.6937315511888]
#reward_sums_apprentice_weighted = [555.5770322738642, 561.3015369017154, 567.318438725923, 558.8736695183811, 565.0696314231911, 558.9010305657155, 556.3769813340348, 556.3772801885348, 580.7108541831584, 551.5193677556914, 551.5014181811501, 552.9761762643988, 556.9237503040824, 579.3869118654817, 553.5396500447647]


# load_apprentice()
# load_apprentice(701) # for window size 7
# agent.policy.approximator.actor.logσ[1] = -14.0f0

batch_size = 20
batch_size_rIC = 100

num_states = 200
num_states_rIC = 4_000


growl_power = 0.09
reweight_power = 0.00004

growl_power_rIC = 0.025
reweight_power_rIC = 0.0001

loss_stop_threshold = 0.001
loss_stop_threshold_rIC = 0.03


growl_freq = 1
growl_srate = 0.999
# theta_rate = 0.7



group_rows_by_overlap = true
group_channels = true

training_steps = 8_000
extra_steps = 0






customCrossAttention = true
jointPPO = false
one_by_one_training = false
positional_encoding = 3 #ZeroEncoding

joon_pe = true
new_pe = false
square_rewards = false
randomIC = true


if randomIC
    block_num = 1
    dim_model = 44
    head_num = 2
    head_dim = 22
    ffn_dim = 44
    drop_out = 0.00#1

    learning_rate = 2e-4
    clip_grad = Inf
else
    block_num = 1
    dim_model = 32
    head_num = 2
    head_dim = 16
    ffn_dim = 32
    drop_out = 0.00#1

    learning_rate = 1e-4
    clip_grad = Inf
end


betas = (0.9, 0.999)

# Tracks which apprentice variant should be persisted/loaded.
apprentice_training_kind = :growl
#apprentice_training_kind = :weighted
apprentice_training_rIC = randomIC

apprentice_agent = create_agent_mat(n_actors = actuators,
                    action_space = actionspace,
                    state_space = env.state_space,
                    use_gpu = false, 
                    rng = rng,
                    y = y, p = p,
                    start_steps = start_steps, 
                    start_policy = start_policy,
                    update_freq = update_freq,
                    learning_rate = learning_rate,
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

    rIC_label = randomIC ? "Variying IC" : "Fixed IC"
    kind_label = apprentice_training_kind == :growl ? "Group Ordered" : "Group Reweighted"
    channels_label = group_channels ? "Grouped Channels" : "Separate Channels"
    trace_name = "Apprentice ($(kind_label), $(channels_label), $(rIC_label))"

    first_layer_matrix = apprentice.encoder.embedding.weight

    back_projection = zeros(size(env.state[:,1]))

    for i in 1:size(first_layer_matrix)[1]
        for j in 1:size(first_layer_matrix)[2]
            back_projection[j] += abs(first_layer_matrix[i,j])
        end
    end

    if new_pe
        channel_size = sensors[2]+1
    else
        channel_size = sensors[2]
    end

    temp_y = reshape(back_projection .* mask, 3,window_size,channel_size)

    p = make_subplots(rows=1, cols=3)

    add_trace!(p, heatmap(z=temp_y[1,:,:]', coloraxis="coloraxis"), col = 1)
    add_trace!(p, heatmap(z=temp_y[2,:,:]', coloraxis="coloraxis"), col = 2)
    add_trace!(p, heatmap(z=temp_y[3,:,:]', coloraxis="coloraxis"), col = 3)

    colorscale = [[0, "rgb(0, 0, 0)"], [0.01, "rgb(59, 24, 124)"], [1, "rgb(195, 131, 255)"], ]

    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 0, cmax = 5, colorscale = colorscale),
            title = trace_name,
            template="plotly_white",
        )


    relayout!(p, layout.fields)

    display(p)

    # now plot the overlayed windows of all agents and the sensor counts by channels

    sensor_window = reshape(mask, 3, window_size, channel_size)
    total_sensors = zeros(3, sensors[1], channel_size)
    window_half_size = Int(floor(window_size/2))

    for i in actuators_to_sensors
        temp_indexes = [(i + j + sensors[1] - 1) % sensors[1] + 1 for j in 0-window_half_size:0+window_half_size]

        total_sensors[:, temp_indexes, :] .+= sensor_window
    end

    total_sensors = clamp.(total_sensors, 0.0f0, 1.0f0)
    total_sensors_combined = total_sensors[1,:,:] + total_sensors[2,:,:] + total_sensors[3,:,:]

    p = plot(heatmap(z=total_sensors_combined', coloraxis="coloraxis"), layout)
    display(p)


    println("--- $trace_name ---")

    indexes_zero = findall(x -> x == 0.0, mask)
    println("Window Sparsity: $(100*length(indexes_zero)/length(mask))%")

    window_sensors_combined = sensor_window[1,:,:] + sensor_window[2,:,:] + sensor_window[3,:,:]
    window_combined = window_sensors_combined[:]
    indexes_zero_combined = findall(x -> x == 0.0, window_combined)
    println("Window Sparsity combined channels: $(100*length(indexes_zero_combined)/length(window_combined))%")

    combined = total_sensors_combined[:]
    indexes_zero_total_combined = findall(x -> x == 0.0, combined)
    println("Total Sparsity combined channels: $(100*length(indexes_zero_total_combined)/length(combined))%")
end

function report_masked_input(; rIC = randomIC, use_evaluation_set::Bool = false)
    global mask

    rIC_label = rIC ? "Variying IC" : "Fixed IC"
    kind_label = apprentice_training_kind == :growl ? "Group Ordered" : "Group Reweighted"
    channels_label = group_channels ? "Grouped Channels" : "Separate Channels"
    trace_name = "Apprentice ($(kind_label), $(channels_label), $(rIC_label))"

    if new_pe
        channel_size = sensors[2] + 1
    else
        channel_size = sensors[2]
    end

    sensor_window = reshape(mask, 3, window_size, channel_size)
    total_sensors = zeros(3, sensors[1], channel_size)
    window_half_size = Int(floor(window_size / 2))

    for i in actuators_to_sensors
        temp_indexes = [(i + j + sensors[1] - 1) % sensors[1] + 1 for j in 0-window_half_size:0+window_half_size]
        total_sensors[:, temp_indexes, :] .+= sensor_window
    end

    total_sensors = clamp.(total_sensors, 0.0f0, 1.0f0)
    total_sensors_combined = total_sensors[1, :, :] + total_sensors[2, :, :] + total_sensors[3, :, :]

    println("--- $trace_name ---")

    indexes_zero = findall(x -> x == 0.0, mask)
    println("Window Sparsity: $(100 * length(indexes_zero) / length(mask))%")

    window_sensors_combined = sensor_window[1, :, :] + sensor_window[2, :, :] + sensor_window[3, :, :]
    window_combined = window_sensors_combined[:]
    indexes_zero_combined = findall(x -> x == 0.0, window_combined)
    println("Window Sparsity combined channels: $(100 * length(indexes_zero_combined) / length(window_combined))%")

    combined = total_sensors_combined[:]
    indexes_zero_total_combined = findall(x -> x == 0.0, combined)
    println("Total Sparsity combined channels: $(100 * length(indexes_zero_total_combined) / length(combined))%")

    if rIC
        if use_evaluation_set
            ensure_rIC_evaluation_cache()
            state_set = rIC_eval_states
            expert_actions = rIC_eval_expert_actions
            dataset_label = "rIC evaluation"
        else
            if !(@isdefined states_rIC)
                grab_states_rIC()
            end
            state_set = states_rIC
            expert_actions = prob(agent.policy, state_set, nothing).μ
            dataset_label = "rIC training"
        end
    else
        if !(@isdefined states)
            generate_states()
        end
        state_set = states
        expert_actions = prob(agent.policy, state_set, nothing).μ
        dataset_label = "fixedIC"
    end

    apprentice_actions = prob(apprentice, state_set .* mask, nothing).μ
    action_diff = apprentice_actions .- expert_actions

    mean_l1_error = mean(abs, action_diff)
    println("Mean L1 error ($(dataset_label) state set): $(mean_l1_error)")

    return mean_l1_error
end

function report_masked_input_all_combinations(; rIC = randomIC, number = nothing, use_evaluation_set::Bool = false)
    global apprentice_training_kind
    global group_channels
    global row_groups

    original_kind = apprentice_training_kind
    original_group_channels = group_channels

    kinds = (:growl, :weighted)
    channel_options = (true, false)
    first_block = true

    try
        if rIC && use_evaluation_set
            ensure_rIC_evaluation_cache()
        end

        for kind in kinds
            for group_channels_value in channel_options
                first_block || println("")
                first_block = false

                apprentice_training_kind = kind
                group_channels = group_channels_value
                row_groups = get_row_groups(group_channels = group_channels)

                load_apprentice(number)
                report_masked_input(rIC = rIC, use_evaluation_set = use_evaluation_set)
            end
        end
    finally
        apprentice_training_kind = original_kind
        group_channels = original_group_channels
        row_groups = get_row_groups(group_channels = group_channels)
    end

    return nothing
end

function build_rIC_evaluation_cache(; steps_per_offset = 200)
    global rIC_eval_states
    global rIC_eval_expert_actions
    global rIC_eval_offsets
    global rIC_eval_steps_per_offset

    steps_per_offset > 0 || error("steps_per_offset must be positive, got $steps_per_offset")

    total_states = length(rIC_validation_offsets) * steps_per_offset
    rIC_eval_states = zeros(Float32, size(env.state)[1], size(env.state)[2], total_states)

    idx = 1
    for offset in rIC_validation_offsets
        println("Generating rIC evaluation data with offset $offset")
        RL.reset!(env)
        generate_random_init(offset)

        for _ in 1:steps_per_offset
            rIC_eval_states[:, :, idx] .= env.state

            action = prob(agent.policy, env.state, nothing).μ
            env(action)

            idx += 1
        end
    end

    rIC_eval_expert_actions = prob(agent.policy, rIC_eval_states, nothing).μ
    rIC_eval_offsets = copy(rIC_validation_offsets)
    rIC_eval_steps_per_offset = steps_per_offset

    println("Prepared rIC evaluation cache: $(total_states) states ($(length(rIC_validation_offsets)) offsets x $(steps_per_offset) steps).")
end

function ensure_rIC_evaluation_cache(; steps_per_offset = 200)
    has_cache = @isdefined(rIC_eval_states) &&
                @isdefined(rIC_eval_expert_actions) &&
                @isdefined(rIC_eval_offsets) &&
                @isdefined(rIC_eval_steps_per_offset)

    expected_total_states = length(rIC_validation_offsets) * steps_per_offset
    cache_matches = has_cache &&
                    size(rIC_eval_states) == (size(env.state)[1], size(env.state)[2], expected_total_states) &&
                    size(rIC_eval_expert_actions, ndims(rIC_eval_expert_actions)) == expected_total_states &&
                    rIC_eval_offsets == rIC_validation_offsets &&
                    rIC_eval_steps_per_offset == steps_per_offset

    if cache_matches
        return
    end

    build_rIC_evaluation_cache(; steps_per_offset = steps_per_offset)
end


function generate_states()
    global states

    states = zeros(Float32, size(env.state)[1], size(env.state)[2], num_states)

    reset!(env)
    generate_random_init()

    for i in 1:num_states

        #action = agent(env)
        action = prob(agent.policy, env.state, nothing).μ

        states[:, :, i] .= env.state

        env(action)

        # println(i, "% of simulation done")
    end
end

function states_rIC_cache_path()
    return joinpath(@__DIR__, "..", "randomIC", "states_rIC_MAT_$(num_states_rIC).jld2")
end

function grab_states_rIC()
    global states_rIC
    cache_path = states_rIC_cache_path()

    if isfile(cache_path)
        loaded_states_rIC = FileIO.load(cache_path, "states_rIC")
        expected_size = (size(env.state)[1], size(env.state)[2], num_states_rIC)
        if size(loaded_states_rIC) == expected_size
            states_rIC = loaded_states_rIC
            println("Loaded states_rIC from cache: $(cache_path)")
            return
        end
        println("Found states_rIC cache with mismatched size, regenerating: $(cache_path)")
    end

    states_rIC = zeros(Float32, size(env.state)[1], size(env.state)[2], num_states_rIC)

    stop_condition = StopAfterEpisodeWithMinSteps(num_states_rIC)

    i = 1
    is_stop = false
    while !is_stop
        reset!(env)
        generate_random_init()

        while !(is_terminated(env) || is_truncated(env))

            action = prob(agent.policy, env.state, nothing).μ

            # take the explorative action here
            # action = agent(env)

            states_rIC[:, :, i] .= deepcopy(env.state)
            i += 1
            if i > num_states_rIC
                is_stop = true
                break
            end

            env(action)

            if stop_condition(agent, env)
                is_stop = true
                break
            end
        end
    end

    cache_dir = dirname(cache_path)
    isdir(cache_dir) || mkdir(cache_dir)
    FileIO.save(cache_path, "states_rIC", states_rIC)
    println("Saved states_rIC cache to: $(cache_path)")
end

function train_apprentice(;mode = apprentice_training_kind, training_steps = training_steps, extra_steps = extra_steps, prune = true, group_rows_by_overlap = group_rows_by_overlap, group_channels = group_channels, rIC = randomIC, weight_update = 10)

    kind_sym = mode isa Symbol ? mode : Symbol(lowercase(string(mode)))
    kind_sym in (:growl, :weighted) || error("mode must be :growl or :weighted")
    use_weighted = kind_sym == :weighted

    global states
    global states_rIC

    if rIC
        if !(@isdefined states_rIC)
            grab_states_rIC()
        end
    else
        if !(@isdefined states)
            generate_states()
        end
    end

    global row_groups
    row_groups = get_row_groups(;group_channels = group_channels)

    global apprentice_training_kind = kind_sym
    global apprentice_training_rIC = rIC

    if use_weighted
        if group_rows_by_overlap
            groups = deepcopy(row_groups)
        else
            n_rows = size(transpose(apprentice.encoder.embedding.weight), 1)
            groups = [[i] for i in 1:n_rows]
        end

        n_groups = length(groups)
        weight_eltype = eltype(apprentice.encoder.embedding.weight)
        global operator_weights = ones(weight_eltype, n_groups)
    end

    report_every = 10
    stop_threshold = rIC ? loss_stop_threshold_rIC : loss_stop_threshold
    threshold_reached_once = false
    stop_training = false

    global losses = Float32[]
    for i in 1:training_steps+extra_steps
        
        i%100 == 0 && i <= training_steps && println(i*100/training_steps, "% done")

        i == training_steps+1 && println("training_steps finished, starting extra_steps...")
        i%100 == 0 && i > training_steps && println((i-training_steps)*100/extra_steps, "% of extra steps done")


        # training call
        current_batch_size = rIC ? batch_size_rIC : batch_size
        num_batches = rIC ? cld(200, current_batch_size) : div(num_states, current_batch_size)

        if use_weighted
            current_power = rIC ? reweight_power_rIC : reweight_power
        else
            current_power = rIC ? growl_power_rIC : growl_power
        end

        if rIC
            rand_inds = shuffle!(rng, Vector(1:num_states_rIC))
        else
            rand_inds = shuffle!(rng, Vector(1:num_states))
        end

        temp_losses = Float32[]

        for j in 1:num_batches
            #println("j is $(j) of $(num_batches)")

            if rIC
                global batch = states_rIC[:, :, rand_inds[(j-1)*current_batch_size+1:j*current_batch_size]]
            else
                global batch = states[:, :, rand_inds[(j-1)*current_batch_size+1:j*current_batch_size]]
            end

            batch_masked = batch .* mask

            na = size(apprentice.decoder.embedding.weight)[2]

            global g_encoder
            global g_decoder

            g_encoder, g_decoder = Flux.gradient(apprentice.encoder, apprentice.decoder) do p_encoder, p_decoder

                obsrep, val = p_encoder(batch_masked)

                # μ, logσ = p_decoder(zeros(Float32,na,1,current_batch_size), obsrep[:,1:1,:])

                # for n in 2:apprentice.n_actors
                #     newμ, newlogσ = p_decoder(cat(zeros(Float32,na,1,current_batch_size), μ, dims=2), obsrep[:,1:n,:])

                #     μ = cat(μ, newμ[:,end:end,:], dims=2)
                # end

                # diff = μ - agent.policy.approximator.actor(batch)[1]


                # new variant
                μ_expert = prob(agent.policy, batch, nothing).μ

                temp_act = cat(zeros(Float32,na,1,current_batch_size),μ_expert[:,1:end-1,:],dims=2)
                μ, logσ = p_decoder(temp_act, obsrep)

                diff = μ - μ_expert
                mse = mean(diff.^2)

                Zygote.ignore() do
                    push!(temp_losses, mse)
                end

                mse
            end

            Flux.update!(apprentice.encoder_state_tree, apprentice.encoder, g_encoder)
            Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder)

            if i%growl_freq == 0 && prune && i <= training_steps
                if use_weighted
                    apply_weighted(
                        apprentice.encoder.embedding.weight;
                        group_rows_by_overlap = group_rows_by_overlap,
                        operator_weights = operator_weights,
                        reweight_power_used = current_power,
                    )
                else
                    apply_growl(
                        apprentice.encoder.embedding.weight;
                        group_rows_by_overlap = group_rows_by_overlap,
                        growl_power_used = current_power,
                    )
                end
            end

        end

        if use_weighted && i%weight_update == 0 && prune && i <= training_steps
            reshaped_weight = transpose(apprentice.encoder.embedding.weight)

            if group_rows_by_overlap
                groups = deepcopy(row_groups)
            else
                n_rows = size(reshaped_weight, 1)
                groups = [[idx] for idx in 1:n_rows]
            end

            n2_groups = [norm(reshaped_weight[idxs, :][:], 2) for idxs in groups]
            eps_val = convert(eltype(operator_weights), 1e-3)
            operator_weights .= one(eltype(operator_weights)) ./ (n2_groups .+ eps_val)
        end


        #check current performance of the apprentice
        # if rIC
        #     diff = prob(apprentice, states_rIC .* mask, nothing).μ - prob(agent.policy, states_rIC, nothing).μ
        # else
        #     diff = prob(apprentice, states .* mask, nothing).μ - prob(agent.policy, states, nothing).μ
        # end
        # mse = mean(diff.^2)

        !isempty(temp_losses) && push!(losses, mean(temp_losses))

        if !isempty(losses)
            current_loss = mean(losses[max(1, end-199):end])
            if !threshold_reached_once && current_loss < stop_threshold
                threshold_reached_once = true
                println("Loss dropped below threshold ($(stop_threshold)) at step $(i): $(current_loss)")
            elseif threshold_reached_once && current_loss > stop_threshold
                println("Stopping training at step $(i): loss rose above threshold ($(stop_threshold)) to $(current_loss)")
                update_mask(0.0)
                stop_training = true
            end
        end

        if i%report_every == 0 
            transposed_weights = transpose(apprentice.encoder.embedding.weight)
            n_rows = size(transposed_weights, 1)
            zero_row_idcs = [i for i in 1:n_rows if all(transposed_weights[i, :] .== 0)]
            println("zero inputs: $(length(zero_row_idcs))")

            weight_factor = sum(abs.(apprentice.encoder.embedding.weight))
            println("weight factor: $(weight_factor)")

            loss_mean = mean(losses[max(1, end-99):end])
            println("mean squared error over last 100 steps: $(loss_mean)")
        end

        #keep the zeros if this is the last pruning step
        if i+growl_freq > training_steps
            println("keeping the zeros in the last $(String(kind_sym)) step")
            update_mask(0.0)
        end

        if stop_training
            break
        end
    end

    plot(losses)
end


function growl_train(;training_steps = training_steps, extra_steps = extra_steps, growl = true, group_rows_by_overlap = group_rows_by_overlap, group_channels = group_channels, rIC = randomIC)
    return train_apprentice(
        mode = :growl,
        training_steps = training_steps,
        extra_steps = extra_steps,
        prune = growl,
        group_rows_by_overlap = group_rows_by_overlap,
        group_channels = group_channels,
        rIC = rIC,
    )
end


function reweight_train(;training_steps = training_steps, extra_steps = extra_steps, reweight = true, group_rows_by_overlap = group_rows_by_overlap, group_channels = group_channels, rIC = randomIC, weight_update = 10)
    return train_apprentice(
        mode = :weighted,
        training_steps = training_steps,
        extra_steps = extra_steps,
        prune = reweight,
        group_rows_by_overlap = group_rows_by_overlap,
        group_channels = group_channels,
        rIC = rIC,
        weight_update = weight_update,
    )
end



function get_row_groups(;group_channels = true)

    row_groups = []

    index_array = collect(1:size(env.state[:,1])[1])

    if new_pe
        channel_size = sensors[2]+1
    else
        channel_size = sensors[2]
    end

    index_y = reshape(index_array, 3,window_size,channel_size)

    # create stencil for grouping
    center_point = Int(ceil(window_size/2))
    agent_delta = Int(sensors[1] / actuators)


    anchor_steps = Int(ceil(center_point/agent_delta)+1)
    
    if group_channels
        stencil_index_array = collect(1:agent_delta*(channel_size))
        index_stencil = reshape(stencil_index_array, 1, agent_delta, channel_size)

        anchors = [
            [1,2,3],
            [center_point + (j * agent_delta) for j in -anchor_steps:anchor_steps],
            [1]
        ]
    else
        stencil_index_array = collect(1:3*agent_delta*(channel_size))
        index_stencil = reshape(stencil_index_array, 3, agent_delta, channel_size)

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




function apply_growl(model_weights; group_rows_by_overlap = true, growl_power_used = growl_power)

    pl_srate = growl_srate

    reshaped_weight = transpose(model_weights)

    global row_groups
    global theta_is

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
    # theta_is = ones(n_groups) * 0.2
    # theta_is[Int(floor((1-theta_rate) * n_groups)):end] .= 1.0

    theta_is = [(i-1)/n_groups for i in 1:n_groups]

    # make the parameters smaller in general
    theta_is .*= growl_power_used

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



function apply_weighted(model_weights; group_rows_by_overlap = true, operator_weights::Vector, reweight_power_used = reweight_power)

    pl_srate = growl_srate

    reshaped_weight = transpose(model_weights)

    global row_groups

    if group_rows_by_overlap
        groups = deepcopy(row_groups)
    else
        groups = [[i] for i in 1:size(reshaped_weight, 1)]
    end

    # Compute the L2 norm for each row group.
    n_groups = length(groups)
    n2_groups = [norm(reshaped_weight[i, :][:], 2) for i in groups]

    length(operator_weights) == n_groups || error("operator_weights length must match number of groups.")

    # Apply weighted L1 proximal operator.
    new_n2_groups = prox_weighted_l1(deepcopy(n2_groups), deepcopy(operator_weights .* reweight_power_used))

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



function prox_weighted_l1(z::Vector, mu::Vector)
    length(z) == length(mu) || error("z and mu must have the same length.")
    x = z .- mu
    x = max.(x, zero(eltype(x)))
    return x
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

        #savefig(p, "frames/a$(lpad(string(i), 4, '0')).png"; width=1600, height=800)
        #body!(w,p)

        temp_reward = reward_function(env; returnGlobalNu = true)
        temp_reward = state_Nu(env)
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

function apprentice_save_stem(; group_channels_value = group_channels)
    global apprentice_training_kind
    global apprentice_training_rIC

    method_tag = apprentice_training_kind == :growl ? "growl" :
                 apprentice_training_kind == :weighted ? "weighted" :
                 string(apprentice_training_kind)
    ric_tag = apprentice_training_rIC ? "rIC_true" : "rIC_false"
    group_channels_suffix = group_channels_value ? "" : "_group_channels_false"

    return "MAT_Apprentice_$(method_tag)_$(ric_tag)$(group_channels_suffix)"
end

function apprentice_save_path(number = nothing; group_channels_value = group_channels)
    stem = apprentice_save_stem(; group_channels_value = group_channels_value)
    filename = isnothing(number) ? "$(stem).jld2" : "$(stem)_$(number).jld2"
    return dirpath * "/saves/" * filename
end

function load_apprentice(number = nothing; group_channels_value = group_channels)
    filepath = apprentice_save_path(number; group_channels_value = group_channels_value)
    global apprentice = FileIO.load(filepath, "apprentice")

    try
        global mask = FileIO.load(filepath, "mask")
    catch
        @warn "No mask found in apprentice save. Keeping current mask." filepath
    end
end

function save_apprentice(number = nothing; group_channels_value = group_channels)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    FileIO.save(apprentice_save_path(number; group_channels_value = group_channels_value), "apprentice", apprentice, "mask", mask)
end





function train_masked(use_random_init = randomIC; visuals = false, num_steps = 1600, inner_loops = 5, outer_loops = 25)
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
        hook.generate_random_init = generate_random_init
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

                    # dist = prob(agent.policy, env.state .* mask, nothing)
                    # action = rand.(agent.policy.rng, dist)

                    # if ndims(action) == 2
                    #     log_p = vec(sum(normlogpdf(dist.μ, dist.σ, action), dims=1))
                    # else
                    #     log_p = normlogpdf(dist.μ, dist.σ, action)
                    # end

                    # agent.policy.last_action_log_prob = log_p[:]



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

    #save_apprentice()
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
