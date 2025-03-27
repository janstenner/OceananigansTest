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
using Printf
#using Blink

sensors = (48,8)
actuators = 12
dt = 1.5





scriptname = "RB_AC_$(dt)_$(sensors[2])"

#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end



# env parameters

seed = Int(floor(rand()*100000))

# seed = 172

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
nna_scale = 6.4
nna_scale_critic = 3.2
drop_middle_layer = true
drop_middle_layer_critic = true
fun = gelu
temporal_steps = 1
action_punish = 0#0.002#0.2
delta_action_punish = 0#0.002#0.5
window_size = 15 #47
use_gpu = false
actionspace = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.95f0

start_steps = -1
start_policy = ZeroPolicy(actionspace)

update_freq = 200


learning_rate = 1e-4
n_epochs = 17
n_microbatches = 10
logσ_is_network = false
max_σ = 10000.0f0
entropy_loss_weight = 0.01
actor_loss_weight = 100.0
critic_loss_weight = 0.001
clip_grad = 0.3
target_kl = 0.8
clip1 = false
start_logσ = - 0.4
tanh_end = false
clip_range = 0.05f0


block_num = 1
dim_model = 22
head_num = 2
head_dim = 11
ffn_dim = 22
drop_out = 0.00#1

betas = (0.9, 0.999)

customCrossAttention = true
adaptive_weights = false
jointPPO = false
one_by_one_training = false
positional_encoding = 3 #ZeroEncoding
square_rewards = true
randomIC = false


chebychev_z = false


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
Pr = 0.7

Re = sqrt(Ra/Pr)

ν = 1 / Re
κ = 1 / sqrt(Ra*Pr)


Δb = 1 


if chebychev_z
    chebychev_spaced_z_faces(k) = 2 - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
    grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = chebychev_spaced_z_faces, topology = (Periodic, Flat, Bounded))

    inner_dt = 0.00012
else
    grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))

    inner_dt = 0.03
end

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

if chebychev_z
    values = FileIO.load("RBmodel300_chebychev.jld2")
else
    values = FileIO.load("RBmodel300.jld2")
end

uu = values["u/data"][4:Nx+3,:,4:Nz+3]
ww = values["w/data"][4:Nx+3,:,4:Nz+4]
bb = values["b/data"][4:Nx+3,:,4:Nz+3]


if randomIC
    circshift_amount = rand(1:Nx)

    uu = circshift(uu, (circshift_amount,0,0))
    ww = circshift(ww, (circshift_amount,0,0))
    bb = circshift(bb, (circshift_amount,0,0))
end

set!(model, u = uu, w = ww, b = bb)

simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
simulation.verbose = false

y0 = zeros(3,Nx,Nz)

y0[1,:,:] = model.tracers.b[1:Nx,1,1:Nz]
y0[2,:,:] = model.velocities.w[1:Nx,1,1:Nz]
y0[3,:,:] = model.velocities.u[1:Nx,1,1:Nz]

y0 = Float32.(y0)


if chebychev_z
    wizard = TimeStepWizard(cfl = 2.4e-2, max_change = 1.00001, max_Δt = 0.007, min_Δt = 0.8 * inner_dt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
    
    start_time = time_ns()
    progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                            sim.model.clock.iteration,
                            sim.model.clock.time,
                            prettytime(1e-9 * (time_ns() - start_time)),
                            sim.Δt,
                            AdvectiveCFL(sim.Δt)(sim.model))
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    simulation.verbose = true
end





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

function reward_function(env; returnGlobalNu = false)
    H = Lz

    delta_T = Δb

    kappa = model.closure.κ[1]

    den = kappa * delta_T / H

    sensordata = env.y[:,sensor_positions[1],sensor_positions[2]]

    q_1_mean = mean(sensordata[1,:,:] .* sensordata[2,:,:])
    Tx = mean(sensordata[1,:,:]', dims = 2)
    q_2 = kappa * mean(array_gradient(Tx))

    globalNu = (q_1_mean - q_2) / den

    if returnGlobalNu
        return globalNu
    end

    rewards = zeros(actuators)

    hor_inv_probes = Int(sensors[1] / actuators)

    for i in 1:actuators
        # tempstate = env.state[:,i]

        # tempT = tempstate[1:3:length(tempstate)]
        # tempW = tempstate[2:3:length(tempstate)]

        # tempT = reshape(tempT, window_size, sensors[2])
        # tempW = reshape(tempW, window_size, sensors[2])

        # tempT = tempT[Int(actuators/2)*hor_inv_probes : (Int(actuators/2)+1)*hor_inv_probes, :]
        # tempW = tempW[Int(actuators/2)*hor_inv_probes : (Int(actuators/2)+1)*hor_inv_probes, :]

        # q_1_mean = mean(tempT .* tempW)
        # Tx = mean(tempT', dims = 2)
        # q_2 = kappa * mean(array_gradient(Tx))

        # localNu = (q_1_mean - q_2) / den

        # rewards[1,i] = 2.89 - (0.995 * globalNu + 0.005 * localNu)
        rewards[i] = 2.6726 - globalNu
        if square_rewards
            rewards[i] = sign(rewards[i]) * rewards[i]^2
        end
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
    # global sensordata
    sensordata = y[:,sensor_positions[1],sensor_positions[2]]

    # New (!!!) Positional Encoding
    P_Temp = zeros(3, sensors[1])

    for j in 1:sensors[1]
        i_rad = (2*pi/sensors[1])*j
        P_Temp[:,j] .= sin(i_rad)
    end

    sensordata = [sensordata;;; P_Temp]

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

    global agent = create_agent_mat(n_actors = actuators,
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
                fun = gelu,
                clip1 = clip1,
                n_epochs = n_epochs,
                n_microbatches = n_microbatches,
                logσ_is_network = logσ_is_network,
                max_σ = max_σ,
                entropy_loss_weight = entropy_loss_weight,
                actor_loss_weight = actor_loss_weight,
                critic_loss_weight = critic_loss_weight,
                adaptive_weights = adaptive_weights,
                clip_grad = clip_grad,
                target_kl = target_kl,
                start_logσ = start_logσ,
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

    global hook = GeneralHook(min_best_episode = min_best_episode,
                collect_NNA = false,
                generate_random_init = generate_random_init,
                collect_history = false,
                collect_rewards_all_timesteps = false,
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
    
    uu = values["u/data"][4:Nx+3,:,4:Nz+3]
    ww = values["w/data"][4:Nx+3,:,4:Nz+4]
    bb = values["b/data"][4:Nx+3,:,4:Nz+3]

    if randomIC
        circshift_amount = rand(1:Nx)

        uu = circshift(uu, (circshift_amount,0,0))
        ww = circshift(ww, (circshift_amount,0,0))
        bb = circshift(bb, (circshift_amount,0,0))
    end

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

    #save()
end


#train()
#train(;num_steps = 140)
#train(;visuals = true, num_steps = 70)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hookMAT.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agentMAT.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hookMAT$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agentMAT$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hookMAT.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agentMAT.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hookMAT$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agentMAT$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end



function render_run(;use_zeros = false)

    # copyto!(agent.policy.behavior_actor, hook.bestNNA)

    # temp_noise = agent.policy.act_noise
    # agent.policy.act_noise = 0.0

    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps  = -1
    
    temp_update_after = agent.policy.update_freq
    agent.policy.update_freq = 100000

    agent.policy.update_step = 0
    global rewards = Float64[]
    global collected_actions = zeros(200,actuators)
    reward_sum = 0.0

    #w = Window()

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

        if use_zeros
            action = zeros(12)'
        else
            action = agent(env)
        end

        collected_actions[i,:] = action[:]
        env(action)

        result = env.y[1,:,:]
        result_W = env.y[2,:,:]
        result_U = env.y[3,:,:]

        p = make_subplots(rows=1, cols=3)

        add_trace!(p, heatmap(z=result', coloraxis="coloraxis"), col = 1)
        add_trace!(p, heatmap(z=result_W'), col = 2)
        add_trace!(p, heatmap(z=result_U'), col = 3)

        # p = plot(heatmap(z=result', coloraxis="coloraxis"), layout)

        relayout!(p, layout.fields)

        savefig(p, "frames/a$(lpad(string(i), 4, '0')).png"; width=2400, height=800)
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