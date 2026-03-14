

rIC_validation_offsets = [73, 28, 47, 90, 30, 42, 5, 53, 35, 65, 17, 22, 26, 40, 46]




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

    apprentice_kind = :growl
    reward_sums_target = Float64[]

    if use_apprentice
        if @isdefined(apprentice_training_kind)
            if apprentice_training_kind isa Symbol
                apprentice_kind = apprentice_training_kind
            elseif apprentice_training_kind isa AbstractString
                apprentice_kind = Symbol(lowercase(apprentice_training_kind))
            end
        end

        if apprentice_kind == :weighted
            global reward_sums_apprentice_weighted = Float64[]
            reward_sums_target = reward_sums_apprentice_weighted
        else
            # Default to growl for backward compatibility.
            apprentice_kind = :growl
            global reward_sums_apprentice_growl = Float64[]
            reward_sums_target = reward_sums_apprentice_growl
        end

        # Keep legacy variable name available for compatibility.
        global reward_sums_apprentice = reward_sums_target
    else
        global reward_sums = Float64[]
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
    println("Mean reward over random ICs ($(apprentice_kind)): $mean_reward")

    traces = AbstractTrace[]
    if @isdefined(reward_sums) && !isempty(reward_sums)
        push!(traces, box(y=reward_sums, name="Expert", boxpoints="all", quartilemethod="linear", boxmean=true))
    end
    if @isdefined(reward_sums_apprentice_growl) && !isempty(reward_sums_apprentice_growl)
        push!(traces, box(y=reward_sums_apprentice_growl, name="Apprentice (Growl)", boxpoints="all", quartilemethod="linear", boxmean=true))
    end
    if @isdefined(reward_sums_apprentice_weighted) && !isempty(reward_sums_apprentice_weighted)
        push!(traces, box(y=reward_sums_apprentice_weighted, name="Apprentice (Weighted)", boxpoints="all", quartilemethod="linear", boxmean=true))
    end

    if isempty(traces)
        println("No reward sums available for plotting.")
    else
        p = plot(traces)
        display(p)
    end

end
