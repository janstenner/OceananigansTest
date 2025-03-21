using Printf
using Oceananigans
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

Nx = 96
Nz = 64


Δt = 0.03
Δt_snap = 1.5
duration = 300

Ra = 1e4
Pr = 0.7

Re = sqrt(Ra/Pr)

ν = 1 / Re
κ = 1 / sqrt(Ra*Pr)


# Temperature difference between bottom and top plate
Δb = 1 

# Set the amplitude of the random perturbation (kick)
kick = 0.2



actions = rand(num_actuators) * 2 .- 1


chebychev_spaced_z_faces(k) = 2 - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

#grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = chebychev_spaced_z_faces, topology = (Periodic, Flat, Bounded))
grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))


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

function bottom_T(x, t)
    return sin(x)
end

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
simulation = Simulation(model, Δt = Δt, stop_time = Δt_snap)

cur_time = 0.0

simulation.verbose = false



# Now, run the simulation
totalsteps = Int(duration/Δt_snap)
results = zeros(totalsteps+1,Nx,Nz)
results[1,:,:] = model.tracers.b[1:Nx,1,1:Nz]

for i in 1:totalsteps
    #new boundary conditions
    global actions = rand(num_actuators) * 2 .- 1
    #model.tracers.b.boundary_conditions.bottom = ValueBoundaryCondition(bottom_T)

    # simulation = Simulation(model, Δt = Δt, stop_time = Δt_snap*i)
    global simulation.stop_time = Δt_snap*i

    run!(simulation)
    global cur_time += Δt_snap

    # collect T
    results[i+1,:,:] = model.tracers.b[1:Nx,1,1:Nz]

    println(cur_time)
end