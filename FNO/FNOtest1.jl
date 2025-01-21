using LinearAlgebra
using Oceananigans
using IntervalSets
using StableRNGs
using Flux
using Random
using PlotlyJS
using Statistics
using Printf
using Optimisers
using NeuralOperators
using CUDA
using JLD2
using Zygote



#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "frames/*")
    println(io, "saves/*")
    println(io, "compare_frames/*")
    println(io, "video_output/*")
end


# this is without 10-timestep-snapshots

num_actuators = 12

Lx = 2*pi
Lz = 2

Nx = 96
Nz = 64

Δt = 0.03
Δt_snap = 0.09
duration = 24.3

start_steps = 20

Ra = 1e4
Pr = 0.71

Re = sqrt(Ra/Pr)

ν = 1 / Re
κ = 1 / sqrt(Ra*Pr)#Re

Δb = 1

# Set the amplitude of the random perturbation (kick)
kick = 0.2

chebychev_z = false

actions = ones(12)


if chebychev_z
    chebychev_spaced_z_faces(k) = 2 - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
    grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = chebychev_spaced_z_faces, topology = (Periodic, Flat, Bounded))
    Δt = 0.00012
else
    grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
    Δt = 0.03
end


function collate_actions_colin(actions, x, t)

    domain = Lx 

    ampl = 0.75  

    dx = 0.03  

    values = ampl.*actions
    Mean = mean(values)
    K2 = maximum([1.0, maximum(abs.(values .- Mean)) / ampl])


    segment_length = domain/num_actuators

    # determine segment of x
    x_segment = Int(floor(x / segment_length) + 1)

    if x_segment == 1
        T0 = 2 + (ampl * actions[end] - Mean)/K2
    else
        T0 = 2 + (ampl * actions[x_segment - 1] - Mean)/K2
    end

    T1 = 2 + (ampl * actions[x_segment] - Mean)/K2

    if x_segment == num_actuators
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
# xx = collect(LinRange(0,2*pi-0.0000001,1000))

# res = Float64[]

# for x in xx
#     append!(res, bottom_T(x,0))
# end

# plot(scatter(x=xx,y=res))



u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
                                bottom = ValueBoundaryCondition(0))
w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
                                bottom = ValueBoundaryCondition(0))
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(1),
                                bottom = ValueBoundaryCondition(bottom_T))#1+Δb))



fno_input_timesteps = 10

ch=(3,64,64,64,64,64, 128, 3)
modes=(16,16,fno_input_timesteps,)
σ = gelu

Transform = NeuralOperators.FourierTransform
lifting = Conv((1,1,1), ch[1]=>ch[2])
mapping = Chain(OperatorKernel(ch[2] => ch[3], modes, Transform, σ; permuted = true),
                OperatorKernel(ch[3] => ch[4], modes, Transform, σ; permuted = true),
                OperatorKernel(ch[4] => ch[5], modes, Transform, σ; permuted = true),
                OperatorKernel(ch[5] => ch[6], modes, Transform; permuted = true))
project = Chain(Conv((1,1,1), ch[6]=>ch[7], σ),
                Conv((1,1,1), ch[7]=>ch[8]))

dev = gpu_device()

fno = FourierNeuralOperator(lifting, mapping, project) |> dev

rng = Random.default_rng()

#ps, st = Lux.setup(rng, fno) |> dev;
optimizer = Optimisers.Adam(0.001)
state_tree = Flux.setup(optimizer, fno)


function train!(model, state_tree, data; epochs = 10)
    # losses = []
    # tstate = Training.TrainState(model, ps, st, Adam(lr))
    # for _ in 1:epochs, (x, y) in data
    #     _, loss, _, tstate = Training.single_train_step!(AutoZygote(), MSELoss(), (x, y),
    #         tstate)
    #     push!(losses, loss)
    # end
    # return losses

    for i in 1:epochs
        g_fno = Flux.gradient(model) do fno
            result = fno(data[1]) - data[2]
            #result = sum(result, dims=(1,2))
            loss = mean((result).^2)

            ignore() do
                push!(global_losses, Float64.(loss))
            end

            loss
        end

        Flux.update!(state_tree, model, g_fno[1])
    end
end






totalsteps = Int(duration/Δt_snap)

visuals_training = false

colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
layout = Layout(
        plot_bgcolor="#f1f3f7",
        coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
)

if visuals_training
    rm(dirpath * "./frames/", recursive=true, force=true)
    mkdir(dirpath * "./frames/")
end

function load(number = nothing)
    if isnothing(number)
        global fno = JLD2.load(dirpath * "./saves/fno.jld2","fno")
        #global fno = deserialize("saves/fno.dat")
    else
        global fno = JLD2.load(dirpath * "./saves/fno$number.jld2","fno")
        #global fno = deserialize("saves/fno$number.dat")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        jldsave(dirpath * "./saves/fno.jld2"; fno)
        #serialize("saves/fno.dat", fno)
    else
        jldsave(dirpath * "./saves/fno$number.dat.jld2"; fno)
        #serialize("saves/fno$number.dat", fno)
    end
end

#load(3)





function train(training_runs = 8)

    global global_losses = Float64[]

    for run in 1:training_runs
        println("starting training run $(run)...")

        global model

        model = NonhydrostaticModel(; grid,
                advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                tracers = (:b),
                buoyancy = Buoyancy(model=BuoyancyTracer()),
                closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                boundary_conditions = (u = u_bcs, b = b_bcs,),
                coriolis = nothing
        )

        # Set initial conditions
        uᵢ(x, z) = kick * randn()
        wᵢ(x, z) = kick * randn()
        bᵢ(x, z) =  1 + (2 - z) * Δb/2 + kick * randn()

        set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

        # Now, we create a 'simulation' to run the model for a specified length of time
        global simulation = Simulation(model, Δt = Δt, stop_time = Δt_snap)

        cur_time = 0.0

        simulation.verbose = false

        global results = zeros(Float32,Nx,Nz,3,totalsteps-20)     # order (x_1, ... , x_d, ch, batch)

        for i in 1:totalsteps
            #new boundary conditions
            #global actions = rand(num_actuators) * 2 .- 1
            #model.tracers.b.boundary_conditions.bottom = ValueBoundaryCondition(bottom_T)

            # simulation = Simulation(model, Δt = Δt, stop_time = Δt_snap*i)
            global simulation.stop_time = Δt_snap*i

            run!(simulation)
            cur_time += Δt_snap

            if visuals_training
                p = plot(heatmap(z=model.tracers.b[1:Nx,1,1:Nz]', coloraxis="coloraxis"), layout)

                savefig(p, dirpath * "./frames//a$(lpad(string((run-1)*totalsteps+i), 5, '0')).png"; width=1000, height=800)
            end


            if i > start_steps
                results[:,:,1,i-start_steps] = model.tracers.b[1:Nx,1,1:Nz]
                results[:,:,2,i-start_steps] = model.velocities.w[1:Nx,1,1:Nz]
                results[:,:,3,i-start_steps] = model.velocities.u[1:Nx,1,1:Nz]
            end

            println(cur_time)
        end


        n_batches = 30
        batch_size = 8
        global global_losses

        println("starting batches...")
        inputs = zeros(Float32,Nx,Nz,fno_input_timesteps,3,batch_size) 
        outputs = zeros(Float32,Nx,Nz,fno_input_timesteps,3,batch_size) 

        for i in 1:n_batches
            GC.gc(true)
            CUDA.reclaim()
            println(i)
            
            for k in 1:batch_size
                start = rand(1:totalsteps-start_steps-2*fno_input_timesteps)
                inputs[:,:,:,:,k] = permutedims(results[:,:,:,start:start+fno_input_timesteps-1],(1,2,4,3))
                outputs[:,:,:,:,k] = permutedims(results[:,:,:,start+fno_input_timesteps:start+2*fno_input_timesteps-1],(1,2,4,3))
            end

            data = [CuArray(deepcopy(inputs)), CuArray(deepcopy(outputs))];

            train!(fno, state_tree, data; epochs=3)
        end
        
    end

    display(plot(global_losses))

    save(3)

    GC.gc(true)
    CUDA.reclaim()
end












function compare(one_by_one = false)

    model = NonhydrostaticModel(; grid,
                advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                tracers = (:b),
                buoyancy = Buoyancy(model=BuoyancyTracer()),
                closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                boundary_conditions = (u = u_bcs, b = b_bcs,),
                coriolis = nothing
    )

    # Set initial conditions
    uᵢ(x, z) = kick * randn()
    wᵢ(x, z) = kick * randn()
    bᵢ(x, z) =  1 + (2 - z) * Δb/2 + kick * randn()

    set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

    # Now, we create a 'simulation' to run the model for a specified length of time
    simulation = Simulation(model, Δt = Δt, stop_time = Δt_snap)

    cur_time = 0.0

    simulation.verbose = false

    global sim_results = zeros(Float32,Nx,Nz,totalsteps-start_steps-fno_input_timesteps)
    global model_results = zeros(Float32,Nx,Nz,totalsteps-start_steps-fno_input_timesteps)
    global last_steps = zeros(Float32,Nx,Nz,fno_input_timesteps,3)

    for i in 1:start_steps+fno_input_timesteps
        global simulation.stop_time = Δt_snap*i

        run!(simulation)
        cur_time += Δt_snap

        if i > start_steps
            last_steps[:,:,i-start_steps,1] = model.tracers.b[1:Nx,1,1:Nz]
            last_steps[:,:,i-start_steps,2] = model.velocities.w[1:Nx,1,1:Nz]
            last_steps[:,:,i-start_steps,3] = model.velocities.u[1:Nx,1,1:Nz]
        end

        println(cur_time)
    end

    function simulate_rest()
        for i in start_steps+fno_input_timesteps+1:totalsteps
            global simulation.stop_time = Δt_snap*i
        
            run!(simulation)
        
            sim_results[:,:,i-start_steps-fno_input_timesteps] = model.tracers.b[1:Nx,1,1:Nz]
        end
    end

    function model_rest()
        global ps, st, fno
        reshaped_input = CuArray(reshape(last_steps,(Nx,Nz,fno_input_timesteps,3,1)))
        for i in start_steps+fno_input_timesteps+1:fno_input_timesteps:totalsteps
            
            reshaped_input = fno(reshaped_input)
        
            if i-start_steps-1 > totalsteps-start_steps-fno_input_timesteps
                difference = i-start_steps-1-totalsteps+start_steps+fno_input_timesteps
                model_results[:,:,i-start_steps-fno_input_timesteps:end] = Array(reshaped_input)[:,:,1:difference,1,1]
            else
                model_results[:,:,i-start_steps-fno_input_timesteps:i-start_steps-1] = Array(reshaped_input)[:,:,:,1,1]
            end
        end
    end

    function model_rest_obo()
        global ps, st, fno
        reshaped_input = CuArray(reshape(last_steps,(Nx,Nz,fno_input_timesteps,3,1)))
        for i in start_steps+fno_input_timesteps+1:totalsteps
            
            new_reshaped_input = fno(reshaped_input)
        
            model_results[:,:,i-start_steps-fno_input_timesteps] = Array(new_reshaped_input)[:,:,1,1,1]

            reshaped_input = circshift(reshaped_input,(0,0,-1,0,0))
            reshaped_input[:,:,end,:,:] = new_reshaped_input[:,:,1,:,:]
        end
    end

    GC.gc(true)
    CUDA.reclaim()

    @time simulate_rest()

    if one_by_one
        @time model_rest_obo()
    else
        @time model_rest()
    end



    rm(dirpath * "./compare_frames/", recursive=true, force=true)
    mkdir(dirpath * "./compare_frames/")

    for i in 1:size(model_results)[3]
        p = make_subplots(rows=1, cols=2)
        add_trace!(p, heatmap(z=sim_results[:,:,i]', coloraxis="coloraxis"), col = 1)
        add_trace!(p, heatmap(z=model_results[:,:,i]', coloraxis="coloraxis"), col = 2)
        relayout!(p, layout.fields)
        savefig(p, dirpath * "./compare_frames//a$(lpad(string(i), 5, '0')).png"; width=1600, height=800)
    end




    isdir(dirpath * "./video_output") || mkdir(dirpath * "./video_output")
    rm(dirpath * "./video_output/comparison.mp4", force=true)
    #run(`ffmpeg -framerate 16 -i "compare_frames/a%05d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "video_output/comparison.mp4"`)

    run(`ffmpeg -framerate 16 -i $(dirpath * "./compare_frames/a%05d.png") -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac $(dirpath * "./video_output/comparison.mp4")`)

    GC.gc(true)
    CUDA.reclaim()
end