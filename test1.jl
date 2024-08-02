# An example of a 2D gravity current (lock-release problem) using Oceananigans

# Load some standard libraries that we will need
using Printf
using Oceananigans
using PlotlyJS

# First, we need to set some physical parameters for the simulation
# Set the domain size in non-dimensional coordinates
Lx = 2*pi  # size in the x-direction
Lz = 2   # size in the vertical (z) direction 

# Set the grid size
Nx = 96  # number of gridpoints in the x-direction
Nz = 64   # number of gridpoints in the z-direction

# Some timestepping parameters
Δt = 0.02 # maximum allowable timestep 
duration = 300 # The non-dimensional duration of the simulation

# Set the Reynolds number (Re=Ul/ν)
Re = 300000

# Set the change in the non-dimensional buouancy 
Δb = 1 

# Set the amplitude of the random perturbation (kick)
kick = 0.05


chebychev_spaced_z_faces(k) = - Lz/2 - Lz/2 * cos(π * (k - 1) / Nz);

# construct a rectilinear grid using an inbuilt Oceananigans function
# Here, the topology parameter sets the style of boundaries in the x, y, and z directions
# 'Bounded' corresponds to wall-bounded directions and 'Flat' corresponds to the dimension that is not considered (here, that is the y direction)
grid = RectilinearGrid(size = (Nx, Nz), x = (0, Lx), z = (0, Lz), topology = (Periodic, Flat, Bounded))



# set the boundary conditions
# FluxBoundaryCondition specifies the momentum or buoyancy flux (in this case zero)
# ValueBoundaryCondition specifies the value of the corresponding variable
# top/bottom correspond to the boundaries in the z-direction
# east/west correspond to the boundaries in the x-direction
# north/south correspond to the boundaries in the y-direction (not used for periodic topology)
# by default, Oceananigans imposes no flux and no normal flow boundary conditions in bounded directions
# hence, we could remove the following lines and get the same result, but we show them here as a demonstration
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0),
                                bottom = FluxBoundaryCondition(0))
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(1+Δb),
                                bottom = ValueBoundaryCondition(1))



# Now, define a 'model' where we specify the grid, advection scheme, bcs, and other settings
model = NonhydrostaticModel(; grid,
              advection = UpwindBiasedFifthOrder(),  # Specify the advection scheme.  Another good choice is WENO() which is more accurate but slower
            timestepper = :RungeKutta3, # Set the timestepping scheme, here 3rd order Runge-Kutta
                tracers = (:b),  # Set the name(s) of any tracers, here b is buoyancy
               buoyancy = Buoyancy(model=BuoyancyTracer()), # this tells the model that b will act as the buoyancy (and influence momentum) 
                closure = (ScalarDiffusivity(ν = 1 / Re, κ = 1 / Re)),  # set a constant kinematic viscosity and diffusivty, here just 1/Re since we are solving the non-dimensional equations 
                boundary_conditions = (u = u_bcs, b = b_bcs), # specify the boundary conditions that we defiend above
               coriolis = nothing # this line tells the mdoel not to include system rotation (no Coriolis acceleration)
)

# Set initial conditions
# Here, we start with a tanh function for buoyancy and add a random perturbation to the velocity. 
uᵢ(x, z) = kick * randn()
wᵢ(x, z) = kick * randn()
bᵢ(x, z) =  1 + ((z+1)/2) * Δb + 0.4 * randn()

# Send the initial conditions to the model to initialize the variables
set!(model, u = uᵢ, w = wᵢ, b = bᵢ)

# Now, we create a 'simulation' to run the model for a specified length of time
simulation = Simulation(model, Δt = Δt, stop_time = duration)

# ### The `TimeStepWizard`
#
# The TimeStepWizard manages the time-step adaptively, keeping the
# Courant-Freidrichs-Lewy (CFL) number close to `1.0` while ensuring
# the time-step does not increase beyond the maximum allowable value
wizard = TimeStepWizard(cfl = 0.85, max_change = 1.1, max_Δt = Δt)
# A "Callback" pauses the simulation after a specified number of timesteps and calls a function (here the timestep wizard to update the timestep)
# To update the timestep more or less often, change IterationInterval in the next line
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# ### A progress messenger
# We add a callback that prints out a helpful progress message while the simulation runs.

start_time = time_ns()

progress(sim) = @printf("i: % 6d, sim time: % 10s, wall time: % 10s, Δt: % 10s, CFL: %.2e\n",
                        sim.model.clock.iteration,
                        sim.model.clock.time,
                        prettytime(1e-9 * (time_ns() - start_time)),
                        sim.Δt,
                        AdvectiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ### Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # extract the buoyancy

# Set the name of the output file
filename = "rayleighbenard"

simulation.output_writers[:xz_slices] =
    JLD2OutputWriter(model, (; u, v, w, b),
                          filename = filename * ".jld2",
                          indices = (:, 1, :),
                         schedule = TimeInterval(1.5),
                            overwrite_existing = true)

# If you are running in 3D, you could save an xy slice like this:                             
#simulation.output_writers[:xy_slices] =
#    JLD2OutputWriter(model, (; u, v, w, b),
#                          filename = filename * "_xy.jld2",
#                          indices = (:,:,10),
#                        schedule = TimeInterval(0.1),
#                            overwrite_existing = true)

nothing # hide

# Now, run the simulation
run!(simulation)


# Plot
