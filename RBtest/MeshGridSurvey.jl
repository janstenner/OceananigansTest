using Printf
using Oceananigans
using FileIO
using JLD2
using PlotlyJS
using Statistics


#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "training_frames/*")
    println(io, "saves/*")
end

num_actuators = 12

Lx = 2*pi
Lz = 2

Nx = [48, 72, 96, 144, 192, 288, 384]
Nz = [32, 48, 64, 96, 128, 192, 256]



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


dt = 1.5


if chebychev_z
    chebychev_spaced_z_faces(k) = 2 - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
    global grid = RectilinearGrid(size = (Nx[1], Nz[1]), x = (0, Lx), z = chebychev_spaced_z_faces, topology = (Periodic, Flat, Bounded))
    inner_dt = 0.00012
else
    global grid = RectilinearGrid(size = (Nx[1], Nz[1]), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
    inner_dt = 0.01
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

# Send the initial conditions to the model to initialize the variables
set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
simulation.verbose = false

if chebychev_z
    wizard = TimeStepWizard(cfl = 2.4e-2, max_change = 1.00001, max_Δt = 0.007, min_Δt = 0.8 * Δt)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
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


function getNu(n)
    result = zeros(3,Nx[n],Nz[n])
    result[1,:,:] = model.tracers.b[1:Nx[n],1,1:Nz[n]]
    result[2,:,:] = model.velocities.w[1:Nx[n],1,1:Nz[n]]
    result[3,:,:] = model.velocities.u[1:Nx[n],1,1:Nz[n]]

    H = Lz

    delta_T = Δb

    kappa = model.closure.κ[1]

    den = kappa * delta_T / H

    q_1_mean = mean(result[1,:,:] .* result[2,:,:])
    Tx = mean(result[1,:,:]', dims = 2)
    q_2 = kappa * mean(array_gradient(Tx))

    globalNu = (q_1_mean - q_2) / den

    
    return globalNu
end





function render_run()

    global runs = []
    #global runs = FileIO.load("runs.jld2","runs")
    global rewards = Float64[]
    global grid                                             
    global model
    global simulation

    rm("frames/", recursive=true, force=true)
    mkdir("frames")

    colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
    ymax = 30
    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
        )

    for n in 1:5

        println("--------------------------")
        println("$(n)")
        println("--------------------------")

        if chebychev_z
            chebychev_spaced_z_faces(k) = 2 - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);
            grid = RectilinearGrid(size = (Nx[n], Nz[n]), x = (0, Lx), z = chebychev_spaced_z_faces, topology = (Periodic, Flat, Bounded))
            inner_dt = 0.00012
        else
            grid = RectilinearGrid(size = (Nx[n], Nz[n]), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
            inner_dt = 0.005
        end

        model = NonhydrostaticModel(; grid,
                    advection = UpwindBiasedFifthOrder(),
                    timestepper = :RungeKutta3,
                    tracers = (:b),
                    buoyancy = Buoyancy(model=BuoyancyTracer()),
                    closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                    boundary_conditions = (u = u_bcs, b = b_bcs,),
                    coriolis = nothing
        )

        # Send the initial conditions to the model to initialize the variables
        set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

        # Now, we create a 'simulation' to run the model for a specified length of time
        simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
        simulation.verbose = false

        if chebychev_z
            wizard = TimeStepWizard(cfl = 2.4e-2, max_change = 1.00001, max_Δt = 0.007, min_Δt = 0.8 * Δt)
            simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
        end

        global rewards = Float64[]

        for i in 1:300


            run!(simulation)

            simulation.stop_time += dt

            p = plot(heatmap(z=model.tracers.b[1:Nx[n],1,1:Nz[n]]', coloraxis="coloraxis"), layout)

            savefig(p, "frames/a$(lpad(string((n-1)*300+i), 4, '0')).png"; width=1000, height=800)

            temp_reward = getNu(n)
            println(temp_reward)

            push!(rewards, temp_reward)

        end

        push!(runs, deepcopy(rewards))
    end


    #two more in 96x64 with inner_dt = 0.03 and inner_dt = 0.0005
    println("--------------------------")
    println("6")
    println("--------------------------")

    grid = RectilinearGrid(size = (Nx[3], Nz[3]), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
    inner_dt = 0.03

    model = NonhydrostaticModel(; grid,
                advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                tracers = (:b),
                buoyancy = Buoyancy(model=BuoyancyTracer()),
                closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                boundary_conditions = (u = u_bcs, b = b_bcs,),
                coriolis = nothing
    )

    set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

    simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
    simulation.verbose = false

    global rewards = Float64[]

    for i in 1:300
        run!(simulation)
        simulation.stop_time += dt
        p = plot(heatmap(z=model.tracers.b[1:Nx[3],1,1:Nz[3]]', coloraxis="coloraxis"), layout)
        savefig(p, "frames/a$(lpad(string(5*300+i), 4, '0')).png"; width=1000, height=800)
        temp_reward = getNu(3)
        println(temp_reward)
        push!(rewards, temp_reward)
    end

    push!(runs, deepcopy(rewards))

    println("--------------------------")
    println("7")
    println("--------------------------")

    grid = RectilinearGrid(size = (Nx[3], Nz[3]), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
    inner_dt = 0.0005

    model = NonhydrostaticModel(; grid,
                advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                tracers = (:b),
                buoyancy = Buoyancy(model=BuoyancyTracer()),
                closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                boundary_conditions = (u = u_bcs, b = b_bcs,),
                coriolis = nothing
    )

    set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

    simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
    simulation.verbose = false

    global rewards = Float64[]

    for i in 1:300
        run!(simulation)
        simulation.stop_time += dt
        p = plot(heatmap(z=model.tracers.b[1:Nx[3],1,1:Nz[3]]', coloraxis="coloraxis"), layout)
        savefig(p, "frames/a$(lpad(string(6*300+i), 4, '0')).png"; width=1000, height=800)
        temp_reward = getNu(3)
        println(temp_reward)
        push!(rewards, temp_reward)
    end

    push!(runs, deepcopy(rewards))

    #one more in 144x96 with inner_dt = 0.0005
    println("--------------------------")
    println("8")
    println("--------------------------")

    grid = RectilinearGrid(size = (Nx[4], Nz[4]), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))
    inner_dt = 0.0005

    model = NonhydrostaticModel(; grid,
                advection = UpwindBiasedFifthOrder(),
                timestepper = :RungeKutta3,
                tracers = (:b),
                buoyancy = Buoyancy(model=BuoyancyTracer()),
                closure = (ScalarDiffusivity(ν = ν, κ = κ)),
                boundary_conditions = (u = u_bcs, b = b_bcs,),
                coriolis = nothing
    )

    set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

    simulation = Simulation(model, Δt = inner_dt, stop_time = dt)
    simulation.verbose = false

    global rewards = Float64[]

    for i in 1:300
        run!(simulation)
        simulation.stop_time += dt
        p = plot(heatmap(z=model.tracers.b[1:Nx[4],1,1:Nz[4]]', coloraxis="coloraxis"), layout)
        savefig(p, "frames/a$(lpad(string(7*300+i), 4, '0')).png"; width=1000, height=800)
        temp_reward = getNu(4)
        println(temp_reward)
        push!(rewards, temp_reward)
    end

    push!(runs, deepcopy(rewards))

    FileIO.save("runs.jld2","runs",runs)

    x_axis = collect(1.5:1.5:450)
    p = plot(scatter(x=x_axis, y=runs[1], name="$(Nx[1]) x $(Nz[1])"))
    for n in 2:5
        add_trace!(p, scatter(x=x_axis, y=runs[n], name="$(Nx[n]) x $(Nz[n])"))
    end
    add_trace!(p, scatter(x=x_axis, y=runs[6], name="$(Nx[3]) x $(Nz[3]), inner_dt = 0.03"))
    add_trace!(p, scatter(x=x_axis, y=runs[7], name="$(Nx[3]) x $(Nz[3]), inner_dt = 0.0005"))
    add_trace!(p, scatter(x=x_axis, y=runs[8], name="$(Nx[4]) x $(Nz[4]), inner_dt = 0.0005"))
    display(p)

    isdir("video_output") || mkdir("video_output")
    rm("video_output/MeshGridSurvey.mp4", force=true)

    run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "video_output/MeshGridSurvey.mp4"`)
end


render_run()