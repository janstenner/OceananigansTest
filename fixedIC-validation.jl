using FileIO

fixedIC_scores_save_default_path = joinpath(@__DIR__, "GrOWL", "saves", "fixedIC_validation_scores.jld2")



function generate_random_init()

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




function same_day_fixed(; use_apprentice = false)

    apprentice_kind = :growl
    group_channels_value = @isdefined(group_channels) ? group_channels : true
    same_day_sum_target = Float64[]

    if use_apprentice
        if @isdefined(apprentice_training_kind)
            if apprentice_training_kind isa Symbol
                apprentice_kind = apprentice_training_kind
            elseif apprentice_training_kind isa AbstractString
                apprentice_kind = Symbol(lowercase(apprentice_training_kind))
            end
        end

        if !@isdefined(same_day_rewards_apprentice_by_config_fixedIC)
            global same_day_rewards_apprentice_by_config_fixedIC = Dict{Tuple{Symbol,Bool}, Vector{Float64}}()
        end

        if apprentice_kind != :weighted
            # Default to growl for backward compatibility.
            apprentice_kind = :growl
        end

        key = (apprentice_kind, group_channels_value)
        same_day_rewards_apprentice_by_config_fixedIC[key] = Float64[]
        same_day_sum_target = same_day_rewards_apprentice_by_config_fixedIC[key]

    else
        global same_day_sum_expert_fixedIC = Float64[]
        same_day_sum_target = same_day_sum_expert_fixedIC
    end


    RL.reset!(env)
    generate_random_init()
    
    
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


    plot_same_day_fixed()

end



function plot_same_day_fixed()

    traces = AbstractTrace[]
    if @isdefined(same_day_sum_expert_fixedIC) && !isempty(same_day_sum_expert_fixedIC)
        push!(traces, scatter(y=same_day_sum_expert_fixedIC, name="Expert"))
    end

    if @isdefined(same_day_rewards_apprentice_by_config_fixedIC)
        config_keys = collect(keys(same_day_rewards_apprentice_by_config_fixedIC))
        sort!(config_keys, by = x -> (x[1] == :growl ? 0 : 1, x[2] ? 0 : 1))

        for (kind, grouped_channels) in config_keys
            y = same_day_rewards_apprentice_by_config_fixedIC[(kind, grouped_channels)]
            isempty(y) && continue

            kind_label = kind == :growl ? "Growl" :
                         kind == :weighted ? "Weighted" :
                         string(kind)
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



function save_fixedIC_scores(filepath = fixedIC_scores_save_default_path)
    isdir(dirname(filepath)) || mkpath(dirname(filepath))


    scores_same_day_expert = @isdefined(reward_sums) ? same_day_sum_expert_fixedIC : Float64[]
    scores_same_day_rewards_apprentice_by_config_fixedIC = @isdefined(same_day_rewards_apprentice_by_config_fixedIC) ? same_day_rewards_apprentice_by_config_fixedIC : Dict{Tuple{Symbol,Bool}, Vector{Float64}}()

    FileIO.save(
        filepath,
        "same_day_sum_expert_fixedIC",scores_same_day_expert,
        "same_day_rewards_apprentice_by_config_fixedIC", scores_same_day_rewards_apprentice_by_config_fixedIC
    )
    println("Saved fixedIC scores to: $(filepath)")
end


function load_fixedIC_scores(filepath = fixedIC_scores_save_default_path)

    global same_day_sum_expert_fixedIC = FileIO.load(filepath, "same_day_sum_expert_fixedIC")
    global same_day_rewards_apprentice_by_config_fixedIC = FileIO.load(filepath, "same_day_rewards_apprentice_by_config_fixedIC")


    println("Loaded fixedIC scores from: $(filepath)")
end


# @show reward_sums
# @show reward_sums_apprentice_growl
# @show reward_sums_apprentice_weighted
