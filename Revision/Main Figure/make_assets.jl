"""
Reproducible vector assets for the main figure. Run from the repository root:
    julia --startup-file=no --project=. "Revision/Main Figure/make_assets.jl"
Only reads saved state and frozen selection artifacts. No simulation or training.
"""
module MainFigureAssets

using JLD2
using JSON
using Printf
using SHA

module ObservationGeometry
    withenv("DISTILLATION_SKIP_AUTOLOAD" => "true") do
        include(joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation", "DistillationCorpus.jl"))
    end
end

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const EXPERIMENT = "260830_231109"
const NX, NZ = 96, 64
const LX, LZ = 2π, 2.0
const SENSOR_X, SENSOR_Z = collect(1:2:95), collect(1:8:57)
const CENTERS = collect(3:4:47)
const CHANNEL_NAMES = ("Temperature", "Vertical velocity", "Horizontal velocity")
const INACTIVE = "#F2F2F2" # Exact Package-7/8 mask-plot color.
const INK, MUTED = "#26364A", "#687587"
const DENSE_STOPS = ((0.0, "#43216F"), (1.0, "#A47ACA"))
const SPARSE_STOPS = ((0.0, "#7E174F"), (1.0, "#D270AB"))
const WINDOW_VARIANTS = (
    (id="03_06_09", title="Drei verteilt", agents=[3,6,9], note="Die bisherige Auswahl mit leicht überlappenden Windows."),
    (id="06", title="Ein Agent", agents=[6], note="Das einzelne Beobachtungsfenster als ruhige Referenz."),
    (id="04_08", title="Zwei verteilt", agents=[4,8], note="Zwei getrennte Windows über dem gleichen Sensorfeld."),
    (id="05_06_07", title="Drei benachbart", agents=[5,6,7], note="Starke Überlappung um die Mitte der Domäne."),
    (id="02_06_10", title="Drei weit verteilt", agents=[2,6,10], note="Fast vollständige Domänenabdeckung mit periodischem Randfenster."),
    (id="01_06_12", title="Drei mit Randübergang", agents=[1,6,12], note="Geteilte Windows an beiden periodischen Rändern."),
    (id="02_05_08_11", title="Vier verteilt", agents=[2,5,8,11], note="Gleichmäßig verteilte Zentren und erkennbare Überlappungen."),
    (id="04_05_06_07", title="Vier benachbart", agents=[4,5,6,7], note="Konzentrierter Ausschnitt mit vier verschobenen Windows."),
    (id="01_03_05_07_09_11", title="Sechs alternierend", agents=[1,3,5,7,9,11], note="Jeder zweite Agent zeigt die wiederkehrende Window-Geometrie."),
    (id="all_12", title="Alle zwölf Agenten", agents=collect(1:12), note="Vollständige Zuordnung aller zwölf Beobachtungsfenster."),
)
# Match the existing RBC sensor/state figure, with its original fixed [1, 2.5] limits.
const TEMPERATURE_STOPS = ((0.0, "#2964BD"), (0.19, "#21A6C4"),
    (0.33, "#F6E6AA"), (0.5, "#DC8C31"), (0.62, "#E34312"), (1.0, "#FC60FF"))
const W = 1152.0
const H = W * LZ / LX

fmt(x::Real) = @sprintf("%.4f", x)
xml(s) = replace(string(s), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;")
file_hash(path) = open(io -> bytes2hex(sha256(io)), path)
relative(path) = replace(relpath(path, ROOT), '\\' => '/')
window_columns(agent) = mod1.(CENTERS[agent] .+ (-7:7), 48)

function parse_arguments(args)
    options = Dict("experiment-id" => EXPERIMENT, "state-file" => joinpath(ROOT, "RBmodel300.jld2"),
        "output-dir" => joinpath(@__DIR__, "assets"), "agents" => "3,6,9")
    check = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--help"
            println("Usage: julia --startup-file=no --project=. \"Revision/Main Figure/make_assets.jl\"\n",
                "  [--experiment-id ID] [--state-file PATH] [--output-dir PATH]\n",
                "  [--agents 3,6,9 | all] [--check-only]\n",
                "Temperature-only assets, ten window selections in two styles, SVG only.\n",
                "--agents changes the default pair and local-window assets; the variation catalogue is always generated.")
            return nothing
        elseif arg == "--check-only"
            check = true
        else
            startswith(arg, "--") || error("Unknown argument: $arg")
            key = arg[3:end]
            haskey(options, key) || error("Unknown option: $arg")
            i < length(args) || error("Missing value after $arg")
            i += 1
            options[key] = args[i]
        end
        i += 1
    end
    agents = options["agents"] == "all" ? collect(1:12) : parse.(Int, split(options["agents"], ','))
    1 <= length(agents) <= 12 && length(unique(agents)) == length(agents) && all(a -> 1 <= a <= 12, agents) ||
        error("--agents requires 1 to 12 distinct indices in 1:12, or all.")
    occursin(r"^[A-Za-z0-9_-]+$", options["experiment-id"]) || error("Invalid experiment ID.")
    return (; experiment=options["experiment-id"], state=abspath(options["state-file"]),
        output=abspath(options["output-dir"]), agents, check)
end

function load_inputs(options)
    selection_path = joinpath(ROOT, "Revision", "Package8", "results", options.experiment,
        "go-gc", "analysis", "selected_test_candidate.jld2")
    saved = JLD2.load(selection_path)
    saved["experiment"] == :package8_varying_regularizer_comparison || error("Not a Package-8 selection.")
    saved["frozen_before_test"] && !saved["selection_uses_test_data"] || error("Selection is not validation-only.")
    candidate = saved["candidate"]
    candidate[:configuration] == "go-gc" || error("Expected GO-GC.")
    candidate[:validation_matching] <= saved["quality_threshold"] || error("Candidate fails frozen validation criterion.")
    mask = BitArray(candidate[:global_mask])
    size(mask) == (3, 48, 8) || error("Expected 3 × 48 × 8 global mask.")
    mask[1,:,:] == mask[2,:,:] == mask[3,:,:] || error("GC channels disagree.")
    count(mask) == candidate[:active_inputs] || error("Active input count mismatch.")
    count(mask[1,:,:]) == candidate[:active_sensor_locations] || error("Active site count mismatch.")
    # Check the global mask against every actual local input row, including periodic windows.
    local_mask = ObservationGeometry.local_mat_observation(Float32.(mask); actuator_sensor_indices=CENTERS)
    all(local_mask[:,a] == candidate[:mask] for a in 1:12) || error("Global/local mask reconstruction mismatch.")
    fields = JLD2.jldopen(options.state, "r") do file
        state = zeros(Float32, 3, NX, NZ)
        # Exact interior slices from FixedIC_MAT.jl, including the native staggered w/u samples.
        for (channel, key) in enumerate(("b/data", "w/data", "u/data"))
            state[channel,:,:] .= file[key][4:NX+3,1,4:NZ+3]
        end
        state
    end
    all(isfinite, fields) || error("State contains nonfinite values.")
    sensors = ObservationGeometry.global_sensor_observation(fields;
        horizontal_indices=SENSOR_X, vertical_indices=SENSOR_Z, add_joon_position_encoding=false)
    local_sensors = ObservationGeometry.local_mat_observation(sensors; actuator_sensor_indices=CENTERS)
    # Independent index audit, rather than relying only on the rendering's reconstruction.
    for agent in 1:12
        cols = window_columns(agent)
        local_sensors[:,agent] == vec(fields[:,SENSOR_X[cols],SENSOR_Z]) || error("Window mapping mismatch.")
    end
    limits = (1.0, 2.5)
    return (; fields, sensors, mask, candidate, saved, selection_path, limits, local_mask)
end

function interpolate_color(value, limits, stops)
    t = clamp((value - limits[1]) / (limits[2] - limits[1]), 0, 1)
    j = min(searchsortedlast([s[1] for s in stops], t), length(stops)-1)
    low, high = stops[j], stops[j+1]
    weight = (t-low[1])/(high[1]-low[1])
    rgb = [round(Int, (1-weight)*parse(Int, low[2][i:i+1]; base=16) +
        weight*parse(Int, high[2][i:i+1]; base=16)) for i in (2,4,6)]
    return @sprintf("#%02X%02X%02X", rgb...)
end
color_for(data, value) = interpolate_color(value, data.limits, TEMPERATURE_STOPS)
# Fixed agent identity across every selection: agent 1 darkest, agent 12 lightest.
agent_color(sparse, agent) = interpolate_color(agent, (1,12), sparse ? SPARSE_STOPS : DENSE_STOPS)

function label(io, x, y, text; size=20, color=INK, anchor="start", weight=400)
    println(io, "<text x=\"$(fmt(x))\" y=\"$(fmt(y))\" font-size=\"$size\" fill=\"$color\" text-anchor=\"$anchor\" font-weight=\"$weight\">$(xml(text))</text>")
end
function rect(io, x, y, w, h; fill="none", stroke="none", sw=1, radius=0, extra="")
    println(io, "<rect x=\"$(fmt(x))\" y=\"$(fmt(y))\" width=\"$(fmt(w))\" height=\"$(fmt(h))\" rx=\"$radius\" fill=\"$fill\" stroke=\"$stroke\" stroke-width=\"$sw\" $extra/>")
end
function line(io, x1, y1, x2, y2; color=INK, sw=1.5, extra="")
    println(io, "<line x1=\"$(fmt(x1))\" y1=\"$(fmt(y1))\" x2=\"$(fmt(x2))\" y2=\"$(fmt(y2))\" stroke=\"$color\" stroke-width=\"$sw\" $extra/>")
end
function write_svg(draw, path, width, height, title, description)
    open(path, "w") do io
        println(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\" width=\"$(fmt(width))\" height=\"$(fmt(height))\" viewBox=\"0 0 $(fmt(width)) $(fmt(height))\" role=\"img\" aria-labelledby=\"title desc\">")
        println(io, "<title id=\"title\">$(xml(title))</title><desc id=\"desc\">$(xml(description))</desc>")
        println(io, "<g font-family=\"Arial, Helvetica, sans-serif\" stroke-linecap=\"round\" stroke-linejoin=\"round\">")
        draw(io)
        println(io, "</g></svg>")
    end
end

# True sample positions: b cell centers; w/u remain sampled as stored by the environment.
# Shared coordinates illustrate the common probe index; native component staggering is not interpolated.
sx(i, x0) = x0 + (SENSOR_X[i]-0.5)/NX*W
sz(j, y0) = y0 + H - (SENSOR_Z[j]-0.5)/NZ*H

function sensor_layer(io, data, sparse; x0=24, y0=24)
    println(io, "<g id=\"sensor-points\">")
    for z in 1:8, x in 1:48
        active = !sparse || data.mask[1,x,z]
        color = active ? color_for(data,data.sensors[1,x,z]) : INACTIVE
        metadata = "data-x-index=\"$x\" data-z-index=\"$z\" data-channel=\"1\" data-represents-channels=\"T,w,u\" data-active=\"$active\""
        title = "Sensor location ($x,$z), representing T/w/u; temperature=$(data.sensors[1,x,z]); " * (active ? "retained" : "masked")
        println(io, "<circle cx=\"$(fmt(sx(x,x0)))\" cy=\"$(fmt(sz(z,y0)))\" r=\"7.5\" fill=\"$color\" $metadata><title>$(xml(title))</title></circle>")
    end
    println(io, "</g>")
end

function contiguous_runs(columns)
    sorted = sort(columns)
    starts = [1; findall(diff(sorted) .> 1) .+ 1]
    stops = [starts[2:end] .- 1; length(sorted)]
    return [(sorted[a],sorted[b]) for (a,b) in zip(starts,stops)]
end

function bracket_lanes(agents)
    # Preserve the staircase of the original three-agent view. For large
    # selections share lanes only between disjoint windows, including seams.
    length(agents) <= 4 && return Dict(a=>i for (i,a) in enumerate(agents))
    occupancy = Set{Int}[]
    lanes = Dict{Int,Int}()
    for a in agents
        cols = Set(window_columns(a))
        lane = findfirst(occupied -> isempty(intersect(cols,occupied)), occupancy)
        if isnothing(lane)
            push!(occupancy,Set{Int}())
            lane = length(occupancy)
        end
        union!(occupancy[lane],cols)
        lanes[a] = lane
    end
    return lanes
end

function window_layer(io, agents, sparse; x0=64, y0=148, style=:frames)
    lanes = bracket_lanes(agents)
    println(io, "<g id=\"agent-windows\">")
    for agent in agents
        color = agent_color(sparse,agent)
        bracket_y = y0-30-32*(lanes[agent]-1)
        println(io,"<g data-agent=\"$agent\" data-agent-color=\"$color\" data-bracket-lane=\"$(lanes[agent])\">")
        for (first_col,last_col) in contiguous_runs(window_columns(agent))
            # Halfway between the adjacent probe columns; clip only at the periodic seam.
            left = max(x0, sx(first_col,x0)-W/96)
            right = min(x0+W, sx(last_col,x0)+W/96)
            if style == :frames
                rect(io,left,y0-10,right-left,H+22;stroke=color,sw=length(agents)>6 ? 1.15 : 1.65,
                    extra="stroke-dasharray=\"6 5\" data-window-first=\"$first_col\" data-window-last=\"$last_col\"")
            end
            line(io,left,bracket_y,right,bracket_y;color,sw=2)
            line(io,left,bracket_y,left,bracket_y+8;color,sw=2)
            line(io,right,bracket_y,right,bracket_y+8;color,sw=2)
            # Label both pieces of a periodic window so each fragment is identifiable.
            center=sx(CENTERS[agent],x0)
            if !(left <= center <= right)
                label(io,(left+right)/2,bracket_y-9,"↔ $agent";size=15,color,anchor="middle")
            end
        end
        label(io,sx(CENTERS[agent],x0),bracket_y-9,"Agent $agent";size=19,color,anchor="middle",weight=600)
        println(io,"</g>")
    end
    println(io,"</g>")
end

function actuator_layer(io, agents, sparse; x0=64, y0=548, show_agents=true)
    println(io,"<g id=\"actuators\">")
    for a in 1:12
        shown = a in agents
        color = shown ? agent_color(sparse,a) : "#CBD1D8"
        rect(io,x0+(a-1)*W/12+2,y0,W/12-4,12;fill=color,radius=2)
        label(io,x0+(a-0.5)*W/12,y0+36,string(a);size=16,color=shown ? color : MUTED,anchor="middle")
    end
    if show_agents
        for a in agents
            color=agent_color(sparse,a)
            center = sx(CENTERS[a],x0)
            line(io,center,y0+44,center,y0+61;color)
            rect(io,center-44,y0+62,88,38;fill="white",stroke=color,sw=1.7,radius=8)
            label(io,center,y0+87,"Agent $a";size=17,color,anchor="middle")
        end
        windows_text=length(agents)==1 ? "1 observation window" : "$(length(agents)) observation windows"
        label(io,x0+W/2,y0+135,"12 actuators · $windows_text · 15 × 8 sensor locations per window";size=19,color=MUTED,anchor="middle")
    end
    println(io,"</g>")
end

function plot_sensors(path, data, sparse, agents; windows=false, style=:frames)
    style in (:frames,:brackets) || error("Unknown window style $style")
    lanes=bracket_lanes(agents)
    x0,y0 = windows ? (64,52+32*maximum(values(lanes))) : (24,24)
    width,height = windows ? (1280, y0+H+178) : (1200,H+48)
    title = (sparse ? "Sparse GO-GC" : "Dense") * " - temperature-colored sensor locations"
    description = "One point represents all three measured channels T/w/u; only temperature determines its color. " *
        "Saved RBC two-plume state; physical probe positions; no positional encoding. " *
        "GO-GC mask from Package 8. Inactive inputs are #F2F2F2. " *
        (windows ? "$(length(agents)) true 15-column observation windows; all 12 actuator segments shown. Style: $style." : "Transparent, independently editable vector elements.")
    write_svg(path,width,height,title,description) do io
        sensor_layer(io,data,sparse;x0,y0)
        if windows
            window_layer(io,agents,sparse;x0,y0,style)
            actuator_layer(io,agents,sparse;x0,y0=y0+H+26)
        end
    end
end

function plot_local_windows(path, data, sparse, agents)
    columns=min(3,length(agents))
    write_svg(path,424*columns+8,410*cld(length(agents),columns),"Local temperature observation windows",
        "$(length(agents)) ordered 15 × 8 windows. Each point represents T/w/u, colored by physical temperature before positional encoding.") do io
        for (panel,agent) in enumerate(agents)
            x0=24+mod(panel-1,columns)*424
            y0=fld(panel-1,columns)*410
            color=agent_color(sparse,agent)
            label(io,x0,y0+27,"Agent $agent";size=24,color,weight=600)
            cols=window_columns(agent)
            retained = sparse ? sum(data.mask[1,cols,:]) : 120
            label(io,x0,y0+55,"15 columns · $retained retained locations";size=16,color=MUTED)
            for z in 1:8, (j,x) in enumerate(cols)
                active=!sparse || data.mask[1,x,z]
                c=active ? color_for(data,data.sensors[1,x,z]) : INACTIVE
                println(io,"<circle cx=\"$(x0+12+(j-1)*26)\" cy=\"$(y0+91+(8-z)*34)\" r=\"8\" fill=\"$c\" data-agent=\"$agent\" data-global-x=\"$x\" data-z-index=\"$z\" data-represents-channels=\"T,w,u\" data-active=\"$active\"/>")
            end
            line(io,x0,y0+350,x0+388,y0+350;color,sw=2)
            label(io,x0,y0+379,"Global columns: $(join(cols, ", "))";size=12,color=MUTED)
        end
    end
end

function plot_field(path,data)
    write_svg(path,1200,H+48,"RBC two-plume temperature field",
        "Full 96 × 64 checkpoint interior; b cell centers; original temperature scale [1, 2.5]. Each cell is a vector rectangle.") do io
        println(io,"<g id=\"temperature-field\" shape-rendering=\"crispEdges\">")
        for z in 1:NZ, x in 1:NX
            rect(io,24+(x-1)*W/NX,24+(NZ-z)*H/NZ,W/NX,H/NZ;
                fill=color_for(data,data.fields[1,x,z]))
        end
        println(io,"</g>")
    end
end

function plot_legend(path,data)
    write_svg(path,1200,110,"Temperature legend",
        "Each point represents all three measured channels at a sensor location; its color shows temperature only.") do io
            label(io,24,43,"Temperature T";size=22,weight=600)
            println(io,"<g shape-rendering=\"crispEdges\">")
            for k in 0:255
                rect(io,215+k*2.5,24,2.5,23;fill=color_for(data,1+1.5*k/255))
            end
            println(io,"</g>")
            for t in (1.0,1.5,2.0,2.5)
                x=215+(t-1)/1.5*640
                line(io,x,50,x,56;color=MUTED)
                label(io,x,79,@sprintf("%.1f",t);size=17,anchor="middle")
            end
            println(io,"<circle cx=\"952\" cy=\"36\" r=\"8\" fill=\"$INACTIVE\"/>")
            label(io,973,43,"Masked sensor";size=20)
    end
end

function plot_agent_palettes(path)
    write_svg(path,1280,198,"Dense and sparse agent colors",
        "Agent identity is consistent across all assets. Dense: purple; sparse: magenta. These colors annotate agents, not temperatures.") do io
        for (row,sparse) in enumerate((false,true))
            y=24+(row-1)*90
            label(io,24,y+23,sparse ? "Sparse" : "Dense";size=23,color=agent_color(sparse,1),weight=600)
            for a in 1:12
                x=150+(a-1)*90
                color=agent_color(sparse,a)
                rect(io,x,y,70,30;fill=color,radius=5)
                label(io,x+35,y+54,"Agent $a";size=15,color,anchor="middle")
            end
        end
    end
end

function plot_controller(path,sparse)
    title=sparse ? "Sparse apprentice" : "Dense expert"
    color=agent_color(sparse,4)
    write_svg(path,600,190,title,"A single shared Multi-Agent Transformer with 12 agent tokens; schematic, not 12 independently trained networks.") do io
        rect(io,12,12,576,166;fill="white",stroke=color,sw=2,radius=15)
        label(io,300,48,title;size=27,color,anchor="middle",weight=600)
        label(io,300,78,"Multi-Agent Transformer · shared parameters";size=18,anchor="middle")
        for a in 1:12
            x=31+(a-1)*45
            token_color=agent_color(sparse,a)
            rect(io,x,97,42,43;fill="white",stroke=token_color,sw=1.5,radius=7)
            label(io,x+21,125,string(a);size=18,color=token_color,anchor="middle")
        end
        label(io,300,162,"12 agent observations → 12 actuator actions";size=16,color=MUTED,anchor="middle")
    end
end

function plot_arrow(path)
    write_svg(path,600,120,"Policy distillation","Dense expert provides action targets for supervised sparse apprentice training with GO-GC regularization.") do io
        label(io,300,27,"Policy distillation";size=25,anchor="middle",weight=600)
        line(io,32,58,555,58;sw=2.5)
        println(io,"<path d=\"M 548 50 L 568 58 L 548 66 Z\" fill=\"$INK\"/>")
        label(io,300,95,"Expert action targets + GO-GC regularization";size=19,color=MUTED,anchor="middle")
    end
end

function write_provenance(path,options,data,files)
    candidate=data.candidate
    provenance=Dict(
        "schema_version"=>2, "iteration"=>2, "experiment_id"=>options.experiment,
        "state_file"=>relative(options.state), "state_sha256"=>file_hash(options.state),
        "state_role"=>"Illustrative saved two-plume initial state from FixedIC_MAT; not a Varying-IC evaluation claim.",
        "selection_file"=>relative(data.selection_path), "selection_sha256"=>file_hash(data.selection_path),
        "candidate_id"=>candidate[:candidate_id], "checkpoint_sha256"=>data.saved["checkpoint_sha256"],
        "run_id"=>candidate[:run_id], "configuration"=>"go-gc", "active_groups"=>candidate[:active_groups],
        "total_groups"=>32, "active_locations"=>count(data.mask[1,:,:]), "total_locations"=>384,
        "active_scalar_inputs"=>count(data.mask), "total_scalar_inputs"=>1152,
        "validation_mse"=>candidate[:validation_matching], "selection_uses_test_data"=>false,
        "regularization_strength"=>candidate[:regularization_strength], "threshold"=>candidate[:threshold_value],
        "active_location_indices_one_based"=>[collect(Tuple(i)) for i in findall(data.mask[1,:,:])],
        "sensor_field_indices"=>Dict("x"=>SENSOR_X,"z"=>SENSOR_Z),
        "domain"=>Dict("Lx"=>LX,"Lz"=>LZ,"Nx"=>NX,"Nz"=>NZ),
        "display_coordinates"=>"x=(field_x_index-0.5)*Lx/Nx; z=(field_z_index-0.5)*Lz/Nz. Common b-cell-center coordinates for grouped probe symbols; u/w retain native sampled values.",
        "channel_order"=>collect(CHANNEL_NAMES), "positional_encoding_displayed"=>false,
        "visualized_channel"=>"temperature", "sensor_point_represents"=>collect(CHANNEL_NAMES),
        "color_limits"=>Dict("temperature"=>collect(data.limits)),
        "agent_colors"=>Dict(prefix=>[agent_color(sparse,a) for a in 1:12] for (prefix,sparse) in (("dense",false),("sparse",true))),
        "window_styles"=>["frames","brackets"],
        "window_variants"=>[Dict("id"=>v.id,"title"=>v.title,"agents"=>v.agents,"note"=>v.note,
            "bracket_lanes"=>[bracket_lanes(v.agents)[a] for a in v.agents]) for v in WINDOW_VARIANTS],
        "inactive_color"=>INACTIVE, "example_agents"=>options.agents,
        "actuator_sensor_centers"=>CENTERS,
        "windows"=>[Dict("agent"=>a,"columns"=>window_columns(a),"active_locations"=>sum(data.mask[1,window_columns(a),:])) for a in 1:12],
        "generator_sha256"=>file_hash(@__FILE__),
        "generator_files_sha256"=>Dict(name=>file_hash(joinpath(@__DIR__,name)) for name in
            ("make_assets.jl","gallery.jl","gallery_header.html")),
        "svg_files"=>sort(files))
    open(path,"w") do io
        JSON.print(io,provenance,2)
        println(io)
    end
end

include(joinpath(@__DIR__,"gallery.jl"))

function main(args=ARGS)
    options=parse_arguments(args)
    isnothing(options) && return
    data=load_inputs(options)
    println("Validated frozen Varying GO-GC candidate $(data.candidate[:candidate_id]): ",
        "$(count(data.mask[1,:,:]))/384 locations, $(count(data.mask))/1152 scalar inputs; all 12 local masks agree.")
    options.check && return
    mkpath(joinpath(options.output,"windows"))
    files=String[]
    asset(name)=(push!(files,name);joinpath(options.output,name))
    for sparse in (false,true)
        prefix=sparse ? "sparse" : "dense"
        plot_sensors(asset("$(prefix)_temperature.svg"),data,sparse,options.agents)
        plot_sensors(asset("$(prefix)_temperature_windows.svg"),data,sparse,options.agents;windows=true)
        plot_local_windows(asset("$(prefix)_local_windows.svg"),data,sparse,options.agents)
        plot_local_windows(asset("$(prefix)_all_local_windows.svg"),data,sparse,collect(1:12))
        plot_controller(asset("$(prefix)_controller.svg"),sparse)
        write_svg(asset("$(prefix)_actuators_12.svg"),1280,78,"12 actuator segments","All twelve agents in the corresponding dense/sparse color family.") do io
            actuator_layer(io,collect(1:12),sparse;x0=64,y0=14,show_agents=false)
        end
        for v in WINDOW_VARIANTS, style in (:frames,:brackets)
            plot_sensors(asset("windows/$(prefix)_$(v.id)_$style.svg"),data,sparse,v.agents;windows=true,style)
        end
    end
    plot_field(asset("two_plume_temperature_field.svg"),data)
    plot_legend(asset("temperature_legend.svg"),data)
    plot_agent_palettes(asset("agent_palettes.svg"))
    plot_arrow(asset("distillation_arrow.svg"))
    write_provenance(joinpath(options.output,"provenance.json"),options,data,files)
    write_gallery(joinpath(options.output,"index.html"),data)
    println("Wrote $(length(files)) temperature-only SVG assets, index.html and provenance.json to $(options.output)")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    MainFigureAssets.main()
end
