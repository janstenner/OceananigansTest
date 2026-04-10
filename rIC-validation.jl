using FileIO

rIC_scores_save_default_path = joinpath(@__DIR__, "GrOWL", "saves", "rIC_validation_scores.jld2")


rIC_validation_offsets = [73, 28, 47, 90, 30, 42, 5, 53, 35, 65, 17, 22, 26, 40, 46]

function normalize_validation_apprentice_kind(kind)::Symbol
    kind_sym = kind isa Symbol ? kind : Symbol(lowercase(string(kind)))
    return kind_sym
end

function validation_apprentice_kind_sort_key(kind)::Tuple{Int, String}
    normalized = normalize_validation_apprentice_kind(kind)
    if normalized == :gro_asc
        return (0, string(normalized))
    elseif normalized == :weighted
        return (1, string(normalized))
    else
        return (2, string(normalized))
    end
end

function validation_apprentice_kind_label(kind)::String
    normalized = normalize_validation_apprentice_kind(kind)
    if @isdefined(apprentice_kind_label)
        return apprentice_kind_label(normalized)
    end

    if normalized == :gro_asc
        return "Group Ordered"
    elseif normalized == :weighted
        return "Group Reweighted"
    else
        return replace(string(normalized), "_" => " ")
    end
end




function generate_random_init(circshift_amount)

    global model = NonhydrostaticModel(; grid,
                advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                tracers = (:b),
                buoyancy = Buoyancy(model=BuoyancyTracer()),
                closure = (ScalarDiffusivity(ν = sqrt(Pr/Ra), κ = 1/sqrt(Pr*Ra))),
                boundary_conditions = (u = u_bcs, b = b_bcs,),
                coriolis = nothing
    )

    global values

    uu = values["u/data"][4:Nx+3,:,4:Nz+3]
    ww = values["w/data"][4:Nx+3,:,4:Nz+4]
    bb = values["b/data"][4:Nx+3,:,4:Nz+3]


    uu = circshift(uu, (circshift_amount,0,0))
    ww = circshift(ww, (circshift_amount,0,0))
    bb = circshift(bb, (circshift_amount,0,0))

    set!(model, u = uu, w = ww, b = bb)


    global simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
    simulation.verbose = false

    result = zeros(3,Nx,Nz)

    result[1,:,:] = model.tracers.b[1:Nx,1,1:Nz]
    result[2,:,:] = model.velocities.w[1:Nx,1,1:Nz]
    result[3,:,:] = model.velocities.u[1:Nx,1,1:Nz]

    env.y0 = Float32.(result)
    env.y = deepcopy(env.y0)
    env.state = env.featurize(; env = env)

    Float32.(result)
end


function validate_agent(; use_apprentice = false)

    apprentice_kind = :gro_asc
    group_channels_value = @isdefined(group_channels) ? group_channels : true
    reward_sums_target = Float64[]

    if use_apprentice
        apprentice_kind = @isdefined(apprentice_training_kind) ? normalize_validation_apprentice_kind(apprentice_training_kind) : :gro_asc

        if !@isdefined(reward_sums_apprentice_by_config)
            global reward_sums_apprentice_by_config = Dict{Tuple{Symbol,Bool}, Vector{Float64}}()
        end

        key = (apprentice_kind, group_channels_value)
        reward_sums_apprentice_by_config[key] = Float64[]
        reward_sums_target = reward_sums_apprentice_by_config[key]

        # Keep legacy variable name available for compatibility.
        global reward_sums_apprentice = reward_sums_target
        if apprentice_kind == :gro_asc && group_channels_value
            global reward_sums_apprentice_gro_asc = reward_sums_target
        elseif apprentice_kind == :growl && group_channels_value
            global reward_sums_apprentice_growl = reward_sums_target
        elseif apprentice_kind == :weighted && group_channels_value
            global reward_sums_apprentice_weighted = reward_sums_target
        end
    else
        global reward_sums = Float64[]
        reward_sums_target = reward_sums
    end

    for j in rIC_validation_offsets
        println("Validating random IC with offset $j")
        RL.reset!(env)
        generate_random_init(j)

        
        reward_sum = 0.0
        
        for i in 1:200

            if use_apprentice
                action = RL.prob(apprentice, env).μ
            else
                action = RL.prob(agent.policy, env).μ
            end

            env(action)

            temp_reward = reward_function(env; returnGlobalNu = true)
            temp_reward = state_Nu(env)
            println(temp_reward)

            reward_sum += temp_reward
        end

        if use_apprentice
            push!(reward_sums_target, reward_sum)
        else
            push!(reward_sums, reward_sum)
        end
    end


    mean_reward = mean(reward_sums_target)
    if use_apprentice
        println("Mean reward over random ICs ($(apprentice_kind), group_channels=$(group_channels_value)): $mean_reward")
    else
        println("Mean reward over random ICs (expert): $mean_reward")
    end

    plot_scores_boxes()

end


function plot_scores_boxes()

    traces = AbstractTrace[]
    if @isdefined(reward_sums) && !isempty(reward_sums)
        push!(traces, box(y=reward_sums, name="Expert", boxpoints="all", quartilemethod="linear", boxmean=true))
    end

    if @isdefined(reward_sums_apprentice_by_config)
        config_keys = collect(keys(reward_sums_apprentice_by_config))
        sort!(config_keys, by = x -> (validation_apprentice_kind_sort_key(x[1]), x[2] ? 0 : 1))

        for (kind, grouped_channels) in config_keys
            y = reward_sums_apprentice_by_config[(kind, grouped_channels)]
            isempty(y) && continue

            kind_label = validation_apprentice_kind_label(kind)
            channels_label = grouped_channels ? "GroupedChannels" : "SeparateChannels"
            trace_name = "Apprentice ($(kind_label), $(channels_label))"

            push!(traces, box(y=y, name=trace_name, boxpoints="all", quartilemethod="linear", boxmean=true))
        end
    else
        # Backward-compatible fallback for older in-memory state.
        if @isdefined(reward_sums_apprentice_gro_asc) && !isempty(reward_sums_apprentice_gro_asc)
            push!(traces, box(y=reward_sums_apprentice_gro_asc, name="Apprentice (Group Ordered)", boxpoints="all", quartilemethod="linear", boxmean=true))
        end
        if @isdefined(reward_sums_apprentice_growl) && !isempty(reward_sums_apprentice_growl)
            push!(traces, box(y=reward_sums_apprentice_growl, name="Apprentice (Group Ordered)", boxpoints="all", quartilemethod="linear", boxmean=true))
        end
        if @isdefined(reward_sums_apprentice_weighted) && !isempty(reward_sums_apprentice_weighted)
            push!(traces, box(y=reward_sums_apprentice_weighted, name="Apprentice (Group Reweighted)", boxpoints="all", quartilemethod="linear", boxmean=true))
        end
    end

    # uncontrolled score (799.2713775861112)
    # push!(traces, box(y=[799.2713775861112], name="Uncontrolled", boxpoints="all", quartilemethod="linear", boxmean=true))

    if isempty(traces)
        println("No reward sums available for plotting.")
    else
        layout = Layout(
            #title="RandomIC reward comparison (MAT vs PPO)",
            #xaxis_title="Step",
            yaxis_title="Cumulative Reward",
            template="plotly_white",
        )

        p = plot(traces, layout)
        display(p)
    end
end



function same_day(; use_apprentice = false)

    apprentice_kind = :gro_asc
    group_channels_value = @isdefined(group_channels) ? group_channels : true
    same_day_sum_target = Float64[]

    if use_apprentice
        apprentice_kind = @isdefined(apprentice_training_kind) ? normalize_validation_apprentice_kind(apprentice_training_kind) : :gro_asc

        if !@isdefined(same_day_rewards_apprentice_by_config)
            global same_day_rewards_apprentice_by_config = Dict{Tuple{Symbol,Bool}, Vector{Float64}}()
        end

        key = (apprentice_kind, group_channels_value)
        same_day_rewards_apprentice_by_config[key] = Float64[]
        same_day_sum_target = same_day_rewards_apprentice_by_config[key]

    else
        global same_day_sum_expert = Float64[]
        same_day_sum_target = same_day_sum_expert
    end

    j = rIC_validation_offsets[1]

    RL.reset!(env)
    generate_random_init(j)
    
    
    for i in 1:200

        if use_apprentice
            action = RL.prob(apprentice, env).μ
        else
            action = RL.prob(agent.policy, env).μ
        end

        env(action)

        temp_reward = state_Nu(env)
        println(temp_reward)

        push!(same_day_sum_target, temp_reward)
    end


    plot_same_day()

end



function plot_same_day()

    traces = AbstractTrace[]
    if @isdefined(same_day_sum_expert) && !isempty(same_day_sum_expert)
        push!(traces, scatter(y=same_day_sum_expert, name="Expert"))
    end

    if @isdefined(same_day_rewards_apprentice_by_config)
        config_keys = collect(keys(same_day_rewards_apprentice_by_config))
        sort!(config_keys, by = x -> (validation_apprentice_kind_sort_key(x[1]), x[2] ? 0 : 1))

        for (kind, grouped_channels) in config_keys
            y = same_day_rewards_apprentice_by_config[(kind, grouped_channels)]
            isempty(y) && continue

            kind_label = validation_apprentice_kind_label(kind)
            channels_label = grouped_channels ? "GroupedChannels" : "SeparateChannels"
            trace_name = "Apprentice ($(kind_label), $(channels_label))"

            push!(traces, scatter(y=y, name=trace_name))
        end
    end

    if isempty(traces)
        println("No reward sums available for plotting.")
    else
        layout = Layout(
            #title="RandomIC reward comparison (MAT vs PPO)",
            xaxis_title="Step",
            yaxis_title="Nu",
            template="plotly_white",
        )
        p = plot(traces, layout)
        display(p)
    end
end



function save_rIC_scores(filepath = rIC_scores_save_default_path)
    isdir(dirname(filepath)) || mkpath(dirname(filepath))

    scores_expert = @isdefined(reward_sums) ? reward_sums : Float64[]
    scores_by_config = @isdefined(reward_sums_apprentice_by_config) ? reward_sums_apprentice_by_config : Dict{Tuple{Symbol,Bool}, Vector{Float64}}()

    scores_same_day_expert = @isdefined(reward_sums) ? same_day_sum_expert : Float64[]
    scores_same_day_rewards_apprentice_by_config = @isdefined(same_day_rewards_apprentice_by_config) ? same_day_rewards_apprentice_by_config : Dict{Tuple{Symbol,Bool}, Vector{Float64}}()

    FileIO.save(
        filepath,
        "reward_sums", scores_expert,
        "reward_sums_apprentice_by_config", scores_by_config,
        "same_day_sum_expert",scores_same_day_expert,
        "same_day_rewards_apprentice_by_config", scores_same_day_rewards_apprentice_by_config
    )
    println("Saved rIC scores to: $(filepath)")
end


function load_rIC_scores(filepath = rIC_scores_save_default_path)
    global reward_sums = FileIO.load(filepath, "reward_sums")
    global reward_sums_apprentice_by_config = FileIO.load(filepath, "reward_sums_apprentice_by_config")

    global same_day_sum_expert = FileIO.load(filepath, "same_day_sum_expert")
    global same_day_rewards_apprentice_by_config = FileIO.load(filepath, "same_day_rewards_apprentice_by_config")

    normalized_scores = Dict{Tuple{Symbol,Bool}, Vector{Float64}}()
    for (key, value) in reward_sums_apprentice_by_config
        normalized_key = (normalize_validation_apprentice_kind(key[1]), key[2])
        normalized_scores[normalized_key] = value
    end
    reward_sums_apprentice_by_config = normalized_scores

    normalized_same_day_scores = Dict{Tuple{Symbol,Bool}, Vector{Float64}}()
    for (key, value) in same_day_rewards_apprentice_by_config
        normalized_key = (normalize_validation_apprentice_kind(key[1]), key[2])
        normalized_same_day_scores[normalized_key] = value
    end
    same_day_rewards_apprentice_by_config = normalized_same_day_scores

    # Backward-compatible aliases for grouped-channel apprentice scores.
    if haskey(reward_sums_apprentice_by_config, (:gro_asc, true))
        global reward_sums_apprentice_gro_asc = reward_sums_apprentice_by_config[(:gro_asc, true)]
    end
    if haskey(reward_sums_apprentice_by_config, (:growl, true))
        global reward_sums_apprentice_growl = reward_sums_apprentice_by_config[(:growl, true)]
    end
    if haskey(reward_sums_apprentice_by_config, (:weighted, true))
        global reward_sums_apprentice_weighted = reward_sums_apprentice_by_config[(:weighted, true)]
    end

    println("Loaded rIC scores from: $(filepath)")
end


# @show reward_sums
# @show reward_sums_apprentice_growl
# @show reward_sums_apprentice_weighted
