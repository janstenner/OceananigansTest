# One deterministic action evaluation at the same state as the main-figure assets.
# Run: julia --startup-file=no --project=. "Revision/Main Figure/make_bottom_temperature_plots.jl"
using RL
using Flux
using JLD2
using JSON
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "make_assets.jl"))
const A = MainFigureAssets
BLAS.set_num_threads(1)

const Lx = 2π
const actuators = 12
actions = zeros(Float32,12)

# Load the actual boundary-function definitions without initializing a simulation.
run_file = joinpath(A.ROOT,"Revision","Run_Files","VaryingIC_MAT.jl")
source = read(run_file,String)
first_function = first(findfirst("function collate_actions_colin",source))
end_functions = first(findnext("# test plot",source,first_function))
include_string(@__MODULE__,source[first_function:prevind(source,end_functions)],run_file)

figure_info = JSON.parsefile(joinpath(@__DIR__,"exports","dense_sparse_all_12_brackets_robots.json"))
state_path = joinpath(A.ROOT,figure_info["state_file"])
selection = JLD2.load(joinpath(A.ROOT,figure_info["selection_file"]))
candidate = selection["candidate"]
candidate_path = joinpath(A.ROOT,"Revision","Package8","results","260830_231109",
    "go-gc","s_0p02","r01","candidates",string(candidate[:checkpoint_id])*".jld2")
expert_path = joinpath(A.ROOT,"Revision","Expert_Apprentice_Distillation","experts","varying","agent.jld2")

fields = JLD2.jldopen(state_path,"r") do file
    values = zeros(Float32,3,96,64)
    for (c,name) in enumerate(("b/data","w/data","u/data"))
        values[c,:,:] .= file[name][4:99,1,4:67]
    end
    values
end
global_input = A.ObservationGeometry.global_sensor_observation(fields;
    horizontal_indices=A.SENSOR_X,vertical_indices=A.SENSOR_Z,add_joon_position_encoding=true)
observation = A.ObservationGeometry.local_mat_observation(global_input;actuator_sensor_indices=A.CENTERS)

println("Loading Dense Expert and selected GO-GC Apprentice...")
expert = JLD2.load(expert_path,"agent").policy
apprentice = JLD2.load(candidate_path,"model_payload")
Flux.testmode!(expert)
Flux.testmode!(apprentice)
dense_actions = vec(Array(RL.prob(expert,observation,nothing).μ))
sparse_actions = vec(Array(RL.prob(apprentice,observation .* Float32.(candidate[:mask]),nothing).μ))

x = collect(range(0,Lx-1e-7;length=2401))
global actions = dense_actions
dense_temperature = [bottom_T(value,0.0) for value in x]
global actions = sparse_actions
sparse_temperature = [bottom_T(value,0.0) for value in x]

function line_plot(path,title,color,x,temperature)
    left,top,width,height = 80.0,45.0,860.0,260.0
    px(value) = left+value/Lx*width
    py(value) = top+(2.8-value)/1.6*height
    A.write_svg(path,1000,370,title,"Deterministic policy mean actions at the displayed two-plume state, passed through bottom_T(x, 0).") do io
        A.label(io,500,28,title;size=21,anchor="middle")
        A.line(io,left,top,left,top+height;color="#444444")
        A.line(io,left,top+height,left+width,top+height;color="#444444")
        for tick in (1.25,1.5,1.75,2.0,2.25,2.5,2.75)
            A.line(io,left,py(tick),left+width,py(tick);color="#E6E6E6",sw=1)
            A.label(io,left-12,py(tick)+5,string(tick);size=14,anchor="end")
        end
        for (tick,text) in ((0.0,"0"),(π/2,"π/2"),(π,"π"),(3π/2,"3π/2"),(2π,"2π"))
            A.line(io,px(tick),top+height,px(tick),top+height+5;color="#444444")
            A.label(io,px(tick),top+height+24,text;size=15,anchor="middle")
        end
        A.label(io,500,356,"x";size=18,anchor="middle")
        println(io,"<text transform=\"translate(20 180) rotate(-90)\" text-anchor=\"middle\" font-size=\"17\">Bottom temperature</text>")
        points = join(("$(A.fmt(px(xi))),$(A.fmt(py(ti)))" for (xi,ti) in zip(x,temperature))," ")
        println(io,"<polyline points=\"$points\" fill=\"none\" stroke=\"$color\" stroke-width=\"2.5\"/>")
    end
end

output = joinpath(@__DIR__,"exports")
mkpath(output)
line_plot(joinpath(output,"dense_expert_bottom_temperature.svg"),"Dense Expert",A.agent_color(false,5),x,dense_temperature)
line_plot(joinpath(output,"go_gc_apprentice_bottom_temperature.svg"),"GO-GC Apprentice",A.agent_color(true,5),x,sparse_temperature)
open(joinpath(output,"bottom_temperature_values.json"),"w") do io
    JSON.print(io,Dict("state_file"=>figure_info["state_file"],"candidate_id"=>candidate[:candidate_id],
        "expert_path"=>expert_path,"apprentice_path"=>candidate_path,"x"=>x,
        "dense_actions"=>dense_actions,"sparse_actions"=>sparse_actions,
        "dense_bottom_temperature"=>dense_temperature,"sparse_bottom_temperature"=>sparse_temperature),2)
end
println("Saved dense_expert_bottom_temperature.svg and go_gc_apprentice_bottom_temperature.svg in $output")
