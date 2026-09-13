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
const CHANNEL_IDS = ("temperature", "vertical_velocity", "horizontal_velocity")
const CHANNEL_COLORS = ("#277DA1", "#F2A13A", "#B41A5C")
const INACTIVE = "#F2F2F2" # Exact Package-7/8 mask-plot color.
const INK, MUTED = "#26364A", "#687587"
const WINDOW_COLORS = ("#5B4A9C", "#167D8D", "#AB4675")
# Match the existing RBC sensor/state figure, with its original fixed [1, 2.5] limits.
const TEMPERATURE_STOPS = ((0.0, "#2964BD"), (0.19, "#21A6C4"),
    (0.33, "#F6E6AA"), (0.5, "#DC8C31"), (0.62, "#E34312"), (1.0, "#FC60FF"))
const VELOCITY_STOPS = ((0.0, "#3264A5"), (0.5, "#F4F0E6"), (1.0, "#B84636"))
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
                "  [--agents 3,6,9] [--check-only]\n",
                "Default: saved two-plume state, frozen Package-8 GO-GC selection, SVG only.")
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
    agents = parse.(Int, split(options["agents"], ','))
    length(agents) == 3 && length(unique(agents)) == 3 && all(a -> 1 <= a <= 12, agents) ||
        error("--agents requires three distinct indices in 1:12.")
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
    limits = [(1.0, 2.5)]
    for channel in 2:3
        limit = max(maximum(abs, sensors[channel,:,:]), eps(Float32))
        push!(limits, (-Float64(limit), Float64(limit)))
    end
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
color_for(data, channel, value) = interpolate_color(value, data.limits[channel],
    channel == 1 ? TEMPERATURE_STOPS : VELOCITY_STOPS)

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

function sensor_layer(io, data, sparse; channel=1, x0=24, y0=24, stripes=false)
    println(io, "<g id=\"sensor-points\">")
    for z in 1:8, x in 1:48
        channels = stripes ? (1:3) : (channel:channel)
        for c in channels
            active = !sparse || data.mask[c,x,z]
            color = active ? (stripes ? CHANNEL_COLORS[c] : color_for(data,c,data.sensors[c,x,z])) : INACTIVE
            metadata = "data-x-index=\"$x\" data-z-index=\"$z\" data-channel=\"$c\" data-active=\"$active\""
            title = "$(CHANNEL_NAMES[c]); sensor ($x,$z); value=$(data.sensors[c,x,z]); " * (active ? "retained" : "masked")
            if stripes
                println(io, "<g $metadata><title>$(xml(title))</title>")
                rect(io, sx(x,x0)-9+(c-1)*6, sz(z,y0)-10, 5.5, 20; fill=color)
                println(io, "</g>")
            else
                println(io, "<circle cx=\"$(fmt(sx(x,x0)))\" cy=\"$(fmt(sz(z,y0)))\" r=\"7.5\" fill=\"$color\" $metadata><title>$(xml(title))</title></circle>")
            end
        end
    end
    println(io, "</g>")
end

function contiguous_runs(columns)
    sorted = sort(columns)
    starts = [1; findall(diff(sorted) .> 1) .+ 1]
    stops = [starts[2:end] .- 1; length(sorted)]
    return [(sorted[a],sorted[b]) for (a,b) in zip(starts,stops)]
end

function window_layer(io, agents; x0=64, y0=148, fill_windows=false)
    println(io, "<g id=\"agent-windows\">")
    for (index, agent) in enumerate(agents)
        color = WINDOW_COLORS[index]
        bracket_y = y0-30-32*(index-1)
        for (first_col,last_col) in contiguous_runs(window_columns(agent))
            # Halfway between the adjacent probe columns; clip only at the periodic seam.
            left = max(x0, sx(first_col,x0)-W/96)
            right = min(x0+W, sx(last_col,x0)+W/96)
            rect(io,left,y0-10,right-left,H+22; fill=fill_windows ? color : "none",
                stroke=color,sw=1.65,extra="fill-opacity=\"0.045\" stroke-dasharray=\"6 5\" data-agent=\"$agent\"")
            line(io,left,bracket_y,right,bracket_y;color,sw=2)
            line(io,left,bracket_y,left,bracket_y+8;color,sw=2)
            line(io,right,bracket_y,right,bracket_y+8;color,sw=2)
        end
        label(io,sx(CENTERS[agent],x0),bracket_y-9,"Agent $agent";size=19,color,anchor="middle",weight=600)
    end
    println(io,"</g>")
end

function actuator_layer(io, agents; x0=64, y0=548, show_agents=true)
    println(io,"<g id=\"actuators\">")
    for a in 1:12
        which = findfirst(==(a),agents)
        color = isnothing(which) ? "#CBD1D8" : WINDOW_COLORS[which]
        rect(io,x0+(a-1)*W/12+2,y0,W/12-4,12;fill=color,radius=2)
        label(io,x0+(a-0.5)*W/12,y0+36,string(a);size=16,color=isnothing(which) ? MUTED : color,anchor="middle")
    end
    if show_agents
        for (index,a) in enumerate(agents)
            center = sx(CENTERS[a],x0)
            line(io,center,y0+44,center,y0+61;color=WINDOW_COLORS[index])
            rect(io,center-47,y0+62,94,38;fill="white",stroke=WINDOW_COLORS[index],sw=1.7,radius=8)
            label(io,center,y0+87,"Agent $a";size=18,color=WINDOW_COLORS[index],anchor="middle")
        end
        label(io,x0+W/2,y0+135,"12 actuators · 3 example observation windows · 15 × 8 probe locations per window";size=19,color=MUTED,anchor="middle")
    end
    println(io,"</g>")
end

function plot_sensors(path, data, sparse, agents; channel=1, windows=false, stripes=false)
    x0,y0 = windows ? (64,148) : (24,24)
    width,height = windows ? (1280, H+326) : (1200,H+48)
    title = (sparse ? "Sparse GO-GC" : "Dense") * " - " * (stripes ? "measurement channels" : CHANNEL_NAMES[channel])
    description = "Saved RBC two-plume state; physical probe positions; no positional encoding. " *
        "GO-GC mask from Package 8. Inactive inputs are #F2F2F2. " *
        (windows ? "Three true 15-column observation windows; all 12 actuator segments shown." : "Transparent, independently editable vector elements.")
    write_svg(path,width,height,title,description) do io
        sensor_layer(io,data,sparse;channel,x0,y0,stripes)
        if windows
            window_layer(io,agents;x0,y0)
            actuator_layer(io,agents;x0,y0=y0+H+26)
        end
    end
end

function plot_local_windows(path, data, sparse, agents)
    write_svg(path,1280,410,"Local temperature observation windows",
        "Three ordered 15 × 8 windows, extracted using the actual periodic local-to-global mapping; physical T before positional encoding.") do io
        for (panel,agent) in enumerate(agents)
            x0=24+(panel-1)*424
            color=WINDOW_COLORS[panel]
            label(io,x0,27,"Agent $agent";size=24,color,weight=600)
            cols=window_columns(agent)
            retained = sparse ? sum(data.mask[1,cols,:]) : 120
            label(io,x0,55,"15 columns · $retained retained locations";size=16,color=MUTED)
            for z in 1:8, (j,x) in enumerate(cols)
                active=!sparse || data.mask[1,x,z]
                c=active ? color_for(data,1,data.sensors[1,x,z]) : INACTIVE
                println(io,"<circle cx=\"$(x0+12+(j-1)*26)\" cy=\"$(91+(8-z)*34)\" r=\"8\" fill=\"$c\" data-agent=\"$agent\" data-global-x=\"$x\" data-z-index=\"$z\" data-active=\"$active\"/>")
            end
            line(io,x0,350,x0+388,350;color,sw=2)
            label(io,x0,379,"Global columns: $(join(cols, ", "))";size=12,color=MUTED)
        end
    end
end

function plot_field(path,data)
    write_svg(path,1200,H+48,"RBC two-plume temperature field",
        "Full 96 × 64 checkpoint interior; b cell centers; original temperature scale [1, 2.5]. Each cell is a vector rectangle.") do io
        println(io,"<g id=\"temperature-field\" shape-rendering=\"crispEdges\">")
        for z in 1:NZ, x in 1:NX
            rect(io,24+(x-1)*W/NX,24+(NZ-z)*H/NZ,W/NX,H/NZ;
                fill=color_for(data,1,data.fields[1,x,z]))
        end
        println(io,"</g>")
    end
end

function plot_legend(path,data;channels=false)
    write_svg(path,1200,110,channels ? "Channel legend" : "Temperature legend",
        "Separate editable legend for the sensor assets.") do io
        if channels
            for c in 1:3
                rect(io,24+(c-1)*295,27,14,25;fill=CHANNEL_COLORS[c])
                label(io,48+(c-1)*295,47,CHANNEL_NAMES[c];size=19)
            end
            rect(io,980,27,14,25;fill=INACTIVE)
            label(io,1004,47,"Masked input";size=19)
        else
            label(io,24,43,"Temperature T";size=22,weight=600)
            println(io,"<g shape-rendering=\"crispEdges\">")
            for k in 0:255
                rect(io,215+k*2.5,24,2.5,23;fill=color_for(data,1,1+1.5*k/255))
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
end

function plot_controller(path,sparse)
    title=sparse ? "Sparse apprentice" : "Dense expert"
    color=sparse ? "#AB4675" : "#277DA1"
    write_svg(path,600,190,title,"A single shared Multi-Agent Transformer with 12 agent tokens; schematic, not 12 independently trained networks.") do io
        rect(io,12,12,576,166;fill="white",stroke=color,sw=2,radius=15)
        label(io,300,48,title;size=27,color,anchor="middle",weight=600)
        label(io,300,78,"Multi-Agent Transformer · shared parameters";size=18,anchor="middle")
        for a in 1:12
            x=31+(a-1)*45
            rect(io,x,97,42,43;fill="white",stroke=color,sw=1.2,radius=7)
            label(io,x+21,125,string(a);size=18,color,anchor="middle")
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
        "schema_version"=>1, "experiment_id"=>options.experiment,
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
        "color_limits"=>Dict(CHANNEL_IDS[c]=>collect(data.limits[c]) for c in 1:3),
        "inactive_color"=>INACTIVE, "example_agents"=>options.agents,
        "actuator_sensor_centers"=>CENTERS,
        "windows"=>[Dict("agent"=>a,"columns"=>window_columns(a),"active_locations"=>sum(data.mask[1,window_columns(a),:])) for a in 1:12],
        "generator_sha256"=>file_hash(@__FILE__), "svg_files"=>sort(files))
    open(path,"w") do io
        JSON.print(io,provenance,2)
        println(io)
    end
end

function write_gallery(path,data)
    pairs=[("Sensorpunkte: Temperatur", "dense_temperature.svg", "sparse_temperature.svg"),
        ("Drei exemplarische Agent-Windows", "dense_temperature_windows.svg", "sparse_temperature_windows.svg"),
        ("Lokale Beobachtungen", "dense_local_windows.svg", "sparse_local_windows.svg"),
        ("Kanalmaske: T / w / u", "dense_channel_mask.svg", "sparse_channel_mask.svg"),
        ("Kanalmaske mit Windows", "dense_channel_mask_windows.svg", "sparse_channel_mask_windows.svg"),
        ("Vertikale Geschwindigkeit", "dense_vertical_velocity.svg", "sparse_vertical_velocity.svg"),
        ("Horizontale Geschwindigkeit", "dense_horizontal_velocity.svg", "sparse_horizontal_velocity.svg"),
        ("Controller-Bausteine", "dense_controller.svg", "sparse_controller.svg")]
    open(path,"w") do io
        println(io,"<!doctype html><html lang=\"de\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Main Figure · Asset-Bibliothek</title><style>body{font:16px Arial,sans-serif;color:#26364a;background:#f3f5f7;margin:0;padding:36px}main{max-width:1500px;margin:auto}h1{font-size:32px;margin-bottom:12px}h2{font-size:21px;margin:38px 0 12px}p{line-height:1.6;max-width:1050px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;background:white;border:1px solid #dde2e8;border-radius:10px;padding:14px}img{width:100%;height:auto}figcaption{margin-top:12px;font-size:14px;color:#687587}a{color:#277da1}.single{margin:18px 0}.single img{max-height:420px} @media(max-width:800px){.pair{grid-template-columns:1fr}body{padding:18px}}</style><main>")
        println(io,"<h1>Main Figure · SVG-Asset-Bibliothek</h1><p>Gespeicherter Two-Plume-Zustand, kombiniert mit der eingefrorenen GO-GC-Maske des Varying-IC-Experiments. Dense: 384 Sensororte / 1152 Eingänge. Sparse: $(count(data.mask[1,:,:])) Sensororte / $(count(data.mask)) Eingänge. Die Temperaturfarben sind identisch; inaktive Punkte sind hellgrau (#F2F2F2).</p><p>Alle SVGs enthalten editierbare Vektorelemente und transparente Hintergründe. Die Messwerte zeigen den physikalischen Zustand vor dem positional encoding. Die Fenster entsprechen den echten 15 × 8 Beobachtungsfenstern; die Controller-Bausteine stehen für einen gemeinsamen MAT mit 12 Agenten.</p>")
        for (heading,left,right) in pairs
            println(io,"<h2>$heading</h2><div class=\"pair\">")
            for file in (left,right)
                println(io,"<figure><a href=\"$file\"><img src=\"$file\" alt=\"$file\"></a><figcaption><a href=\"$file\">$file</a></figcaption></figure>")
            end
            println(io,"</div>")
        end
        println(io,"<h2>Weitere Bausteine</h2>")
        for file in ("two_plume_temperature_field.svg","temperature_legend.svg","channel_legend.svg","actuators_12.svg","distillation_arrow.svg")
            println(io,"<figure class=\"single\"><a href=\"$file\"><img src=\"$file\" alt=\"$file\"></a><figcaption>$file</figcaption></figure>")
        end
        println(io,"<p><a href=\"provenance.json\">Datenherkunft, Maskenidentität, Koordinaten und alle zwölf Window-Zuordnungen</a></p></main></html>")
    end
end

function main(args=ARGS)
    options=parse_arguments(args)
    isnothing(options) && return
    data=load_inputs(options)
    println("Validated frozen Varying GO-GC candidate $(data.candidate[:candidate_id]): ",
        "$(count(data.mask[1,:,:]))/384 locations, $(count(data.mask))/1152 scalar inputs; all 12 local masks agree.")
    options.check && return
    mkpath(options.output)
    files=String[]
    asset(name)=(push!(files,name);joinpath(options.output,name))
    for sparse in (false,true)
        prefix=sparse ? "sparse" : "dense"
        for channel in 1:3
            plot_sensors(asset("$(prefix)_$(CHANNEL_IDS[channel]).svg"),data,sparse,options.agents;channel)
        end
        plot_sensors(asset("$(prefix)_temperature_windows.svg"),data,sparse,options.agents;windows=true)
        plot_sensors(asset("$(prefix)_channel_mask.svg"),data,sparse,options.agents;stripes=true)
        plot_sensors(asset("$(prefix)_channel_mask_windows.svg"),data,sparse,options.agents;stripes=true,windows=true)
        plot_local_windows(asset("$(prefix)_local_windows.svg"),data,sparse,options.agents)
        plot_controller(asset("$(prefix)_controller.svg"),sparse)
    end
    plot_field(asset("two_plume_temperature_field.svg"),data)
    plot_legend(asset("temperature_legend.svg"),data)
    plot_legend(asset("channel_legend.svg"),data;channels=true)
    write_svg(asset("actuators_12.svg"),1280,78,"12 actuator segments","Equal actuator segments; three example agents highlighted.") do io
        actuator_layer(io,options.agents;x0=64,y0=14,show_agents=false)
    end
    plot_arrow(asset("distillation_arrow.svg"))
    write_provenance(joinpath(options.output,"provenance.json"),options,data,files)
    write_gallery(joinpath(options.output,"index.html"),data)
    println("Wrote $(length(files)) SVG assets, index.html and provenance.json to $(options.output)")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    MainFigureAssets.main()
end
