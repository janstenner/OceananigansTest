using LinearAlgebra
using Oceananigans
using RL
using IntervalSets
using StableRNGs
#using SparseArrays
using Flux
using Random
using PlotlyJS
using FileIO, JLD2
using Statistics
#using Blink

sensors = (48,8)
actuators = 12
dt = 1.5
inner_dt = 0.03




scriptname = "RB_AC_$(dt)_$(sensors[2])"

#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end



# env parameters

seed = Int(floor(rand()*1000))

te = 300.0
t0 = 0.0
min_best_episode = 1

check_max_value = "nothing"
max_value = 30.0

Nx = 96
Nz = 64
Lx = 2*pi
Lz = 2
dx  = Lx/Nx;        
dz  = Lz/Nz;
sim_space = Space(fill(0..1, (Nx, Nz)))








# sensor positions - 
variance = 0.001

sensor_positions = [collect(1:Int(Nx/sensors[1]):Nx), collect(1:Int(Nz/sensors[2]):Nz)]

actuator_positions = collect(1:Int(Nx/actuators):Nx)

actuators_to_sensors = [findfirst(x->x==i, sensor_positions[1]) for i in actuator_positions]


# agent tuning parameters
memory_size = 0
nna_scale = 10.0
nna_scale_critic = 7.0
drop_middle_layer = false
drop_middle_layer_critic = false
fun = leakyrelu
temporal_steps = 1
action_punish = 0#0.002#0.2
delta_action_punish = 0#0.002#0.5
window_size = 23
use_gpu = false
actionspace = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99997f0
p = 0.9995f0

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 80


learning_rate = 2e-4
n_epochs = 5
n_microbatches = 16
logσ_is_network = false
max_σ = 10000.0f0
entropy_loss_weight = 0.01
clip_grad = 0.5
target_kl = 0.08
clip1 = false
start_logσ = -0.8
tanh_end = false


actions = rand(actuators) * 2 .- 1


function collate_actions_colin(actions, x, t)

    domain = Lx 

    ampl = 0.75  

    dx = 0.03  

    values = ampl.*actions
    Mean = mean(values)
    K2 = maximum([1.0, maximum(abs.(values .- Mean)) / ampl])


    segment_length = domain/actuators

    # determine segment of x
    x_segment = Int(floor(x / segment_length) + 1)

    if x_segment == 1
        T0 = 2 + (ampl * actions[end] - Mean)/K2
    else
        T0 = 2 + (ampl * actions[x_segment - 1] - Mean)/K2
    end

    T1 = 2 + (ampl * actions[x_segment] - Mean)/K2

    if x_segment == actuators
        T2 = 2 + (ampl * actions[1] - Mean)/K2
    else
        T2 = 2 + (ampl * actions[x_segment + 1] - Mean)/K2
    end

    # x position in the segment
    x_pos = x - (x_segment - 1) * segment_length

    # determine if x is in the transition regions

    if x_pos < dx

        #transition region left
        return T0+((T0-T1)/(4*dx^3)) * (x_pos - 2*dx) * (x_pos + dx)^2

    elseif x_pos >= segment_length - dx

        #transition region right
        return T1+((T1-T2)/(4*dx^3)) * (x_pos - segment_length - 2*dx) * (x_pos - segment_length + dx)^2

    else

        # middle of the segment
        return T1

    end
end

function bottom_T(x, t)
    global actions
    collate_actions_colin(actions,x,t)
end


# test plot
xx = collect(LinRange(0,2*pi-0.0000001,1000))

res = Float64[]

for x in xx
    append!(res, bottom_T(x,0))
end

plot(scatter(x=xx,y=res))



Ra = 1e4
Pr = 0.71

Re = sqrt(Ra/Pr)

ν = 1 / Re
κ = 1 / Re


Δb = 1 

grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))

u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
                                bottom = ValueBoundaryCondition(0))
w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
                                bottom = ValueBoundaryCondition(0))
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(1),
                                bottom = ValueBoundaryCondition(bottom_T))#1+Δb))

model = NonhydrostaticModel(; grid,
              advection = UpwindBiasedFifthOrder(),
              timestepper = :RungeKutta3,
              tracers = (:b),
              buoyancy = Buoyancy(model=BuoyancyTracer()),
              closure = (ScalarDiffusivity(ν = ν, κ = κ)),
              boundary_conditions = (u = u_bcs, b = b_bcs,),
              coriolis = nothing
)

values = FileIO.load("RBmodel300_iteration10001.jld2");

set!(model, u = values["u/data"][4:Nx+3,:,4:Nz+3], w = values["w/data"][4:Nx+3,:,4:Nz+4], b = values["b/data"][4:Nx+3,:,4:Nz+3])

simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
simulation.verbose = false

y0 = zeros(3,Nx,Nz)

y0[1,:,:] = model.tracers.b[1:Nx,1,1:Nz]
y0[2,:,:] = model.velocities.w[1:Nx,1,1:Nz]
y0[3,:,:] = model.velocities.u[1:Nx,1,1:Nz]

y0 = Float32.(y0)








function do_step(env)

    # function bottom_T automatically takes global actions variable into account
    global actions = env.p[:]

    global simulation
    global model

    run!(simulation)

    simulation.stop_time += dt

    result = zeros(3,Nx,Nz)


    result[1,:,:] = model.tracers.b[1:Nx,1,1:Nz]
    result[2,:,:] = model.velocities.w[1:Nx,1,1:Nz]
    result[3,:,:] = model.velocities.u[1:Nx,1,1:Nz]

    result
end


function array_gradient(a)
    result = zeros(length(a))

    for i in 1:length(a)
        if i == 1
            result[i] = a[i+1] - a[i]
        elseif i == length(a)
            result[i] = a[i] - a[i-1]
        else
            result[i] = (a[i+1] - a[i-1]) / 2
        end
    end

    result
end

function reward_function(env)
    H = Lz

    delta_T = 1.0

    kappa = model.closure.κ[1]

    den = kappa * delta_T / H

    q_1_mean = mean(env.y[1,:,:] .* env.y[2,:,:])

    Tx = mean(env.y[1,:,:]', dims = 2)

    q_2 = kappa * mean(array_gradient(Tx))

    globalNu = (q_1_mean - q_2) / den

    rewards = zeros(actuators)

    for i in 1:actuators
        tempstate = env.state[:,i]

        tempT = tempstate[1:3:length(tempstate)]
        tempU1 = tempstate[2:3:length(tempstate)]

        q_1_mean = mean(tempT .* tempU1)

        #Tx = mean(env.y[1,:,:], dims = 2)
        Tx = zeros(sensors[2])

        for j in 1:window_size
            Tx .+= tempT[ 8*(j-1)+1 : j*8 ]
        end

        q_2 = kappa * mean(array_gradient(Tx))

        localNu = (q_1_mean - q_2) / den

        # rewards[1,i] = 2.89 - (0.995 * globalNu + 0.005 * localNu)
        rewards[i] = - (globalNu/2.89)^2
    end
 
    return rewards
end



function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = y0
    else
        y = env.y
    end

    # convolution is delta
    sensordata = y[:,sensor_positions[1],sensor_positions[2]]

    # New Positional Encoding
    P_Temp = zeros(sensors[1], sensors[2])

    for j in 1:sensors[1]
        i_rad = (2*pi/sensors[1])*j
        P_Temp[j,:] .= sin(i_rad)
    end

    sensordata[1,:,:] += P_Temp

    window_half_size = Int(floor(window_size/2))

    result = Vector{Vector{Float64}}()

    for i in actuators_to_sensors
        temp_indexes = [(i + j + sensors[1] - 1) % sensors[1] + 1 for j in 0-window_half_size:0+window_half_size]

        tempresult = sensordata[:,temp_indexes,:]


        push!(result, tempresult[:])
    end

    result = reduce(hcat,result)


    if temporal_steps > 1
        if isnothing(env)
            resulttemp = result
            for i in 1:temporal_steps-1
                result = vcat(result, resulttemp)
            end
        else
            result = vcat(result, env.state[1:end-size(result)[1]-memory_size,:])
        end
    end

    if memory_size > 0
        if isnothing(env)
            result = vcat(result, zeros(memory_size, length(actuator_positions)))
        else
            result = vcat(result, env.action[end-(memory_size-1):end,:])
        end
    end

    return Float32.(result)
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing) 
    if isnothing(env)
        action =  action0
        p = action0
    else
        action = env.action
        p = env.p
    end

    # action = 0.8 * action + 0.2 * p

    return action
end


# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = GeneralEnv(do_step = do_step, 
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = actionspace,
                max_value = max_value,
                check_max_value = check_max_value)

    global agent = create_agent_ppo(n_envs = actuators,
                action_space = actionspace,
                state_space = env.state_space,
                use_gpu = use_gpu, 
                rng = rng,
                y = y, p = p,
                start_steps = start_steps, 
                start_policy = start_policy,
                update_freq = update_freq,
                learning_rate = learning_rate,
                nna_scale = nna_scale,
                nna_scale_critic = nna_scale_critic,
                drop_middle_layer = drop_middle_layer,
                drop_middle_layer_critic = drop_middle_layer_critic,
                fun = fun,
                clip1 = clip1,
                n_epochs = n_epochs,
                n_microbatches = n_microbatches,
                logσ_is_network = logσ_is_network,
                max_σ = max_σ,
                entropy_loss_weight = entropy_loss_weight,
                clip_grad = clip_grad,
                target_kl = target_kl,
                start_logσ = start_logσ,
                tanh_end = tanh_end,)

    global hook = GeneralHook(min_best_episode = min_best_episode,
                collect_NNA = false,
                generate_random_init = generate_random_init,
                collect_history = false,
                collect_rewards_all_timesteps = true,
                early_success_possible = false)
end

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
    set!(model, u = values["u/data"][1:Nx,:,1:Nz], w = values["w/data"][1:Nx,:,1:Nz+1], b = values["b/data"][1:Nx,:,1:Nz])


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

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = true; visuals = false, num_steps = 1600, inner_loops = 5, outer_loops = 1)
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

    save()
end


#train()
#train(;num_steps = 140)
#train(;visuals = true, num_steps = 70)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end



function render_run()

    # copyto!(agent.policy.behavior_actor, hook.bestNNA)

    # temp_noise = agent.policy.act_noise
    # agent.policy.act_noise = 0.0

    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps  = -1
    
    temp_update_after = agent.policy.update_freq
    agent.policy.update_freq = 100000

    agent.policy.update_step = 0
    global rewards = Float64[]
    reward_sum = 0.0

    #w = Window()

    rm("frames/", recursive=true, force=true)
    mkdir("frames")

    colorscale = [[0, "rgb(34, 74, 168)"], [0.5, "rgb(224, 224, 180)"], [1, "rgb(156, 33, 11)"], ]
    ymax = 30
    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
        )


    reset!(env)
    generate_random_init()

    for i in 1:200
        action = agent(env)

        env(action)

        result = env.y[1,:,:]

        p = plot(heatmap(z=result', coloraxis="coloraxis"), layout)

        savefig(p, "frames/a$(lpad(string(i), 4, '0')).png"; width=1000, height=800)
        #body!(w,p)

        temp_reward = env.reward[1]
        println(temp_reward)

        reward_sum += temp_reward
        push!(rewards, temp_reward)

        # println(mean(env.reward))

        # reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))
    end

    println(reward_sum)

    #copyto!(agent.policy.behavior_actor, hook.currentNNA)

    agent.policy.start_steps = temp_start_steps
    agent.policy.update_freq = temp_update_after

    if true
        isdir("video_output") || mkdir("video_output")
        rm("video_output/$scriptname.mp4", force=true)
        #run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "video_output/$scriptname.mp4"`)

        run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "video_output/$scriptname.mp4"`)
    end
end

# t1 = scatter(y=rewards1)
# t2 = scatter(y=rewards2)
# t3 = scatter(y=rewards3)
# plot([t1, t2, t3])