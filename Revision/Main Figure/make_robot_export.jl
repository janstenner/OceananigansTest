"""
Selected main-figure composition: all 12 brackets, dense above sparse,
with the supplied vector robot instead of bottom agent labels and boxes.

Run from the repository root:
    julia --startup-file=no --project=. "Revision/Main Figure/make_robot_export.jl"
"""
module MainFigureRobotExport

include(joinpath(@__DIR__, "make_assets.jl"))
using .MainFigureAssets
using JSON

const A = MainFigureAssets
const OUTPUT_STEM = "dense_sparse_all_12_brackets_robots"
const ROBOT_PATH = joinpath(@__DIR__, "robot.svg")
const ROBOT_HEIGHT = 72.0
const PANEL_GAP = 44.0
const PANEL_WIDTH = 1280.0
const X0 = 64.0
const Y0 = 180.0

function read_robot(path)
    svg = read(path, String)
    root = match(r"<svg\b([^>]*)>(.*)</svg>"s, svg)
    isnothing(root) && error("No SVG root in $path")
    attrs, body = root.captures
    viewbox = match(r"\bviewBox=\"([^\"]+)\"", attrs)
    isnothing(viewbox) && error("The robot SVG needs a viewBox.")
    box = parse.(Float64, split(viewbox[1]))
    length(box) == 4 && all(isfinite,box) && box[3] > 0 && box[4] > 0 || error("Invalid robot viewBox.")
    # This importer is deliberately scoped to the provided, self-contained vector icon.
    occursin(r"<(?:image|script|foreignObject|use)\b|\b(?:href|id)=|url\("i, body) &&
        error("Expected a self-contained robot made from inline vector paths.")
    paths = collect(eachmatch(r"<path\b",body))
    isempty(paths) && error("The supplied robot has no vector paths.")
    style_match = match(r"\bstyle=\"([^\"]*)\"",attrs)
    style = isnothing(style_match) ? "" : style_match[1]
    return (; body, viewbox=viewbox[1], width=box[3], height=box[4], style, path_count=length(paths))
end

function robot_icon(io, robot, prefix, agent, color, center, top)
    width = ROBOT_HEIGHT * robot.width / robot.height
    # Preserve original path geometry, transforms and even-odd cutouts. The icon's
    # gray fills and near-opaque export alpha become the exact agent annotation color.
    body = replace(robot.body, r"\bfill\s*:\s*rgb\([^;]+\)" => "fill:$color")
    body = replace(body, r"\bfill-opacity\s*:\s*[0-9.]+" => "fill-opacity:1")
    occursin("fill:$color",body) || error("No recognized robot fill to recolor.")
    println(io,"<g id=\"$prefix-robot-$agent\" data-agent=\"$agent\" data-agent-color=\"$color\">")
    println(io,"<svg x=\"$(A.fmt(center-width/2))\" y=\"$(A.fmt(top))\" width=\"$(A.fmt(width))\" height=\"$(A.fmt(ROBOT_HEIGHT))\" viewBox=\"$(A.xml(robot.viewbox))\" style=\"$(A.xml(robot.style))\" aria-label=\"Agent $agent\">")
    println(io,body)
    println(io,"</svg></g>")
end

function robot_actuators(io, robot, sparse; top)
    prefix = sparse ? "sparse" : "dense"
    println(io,"<g id=\"$prefix-actuators-and-robots\">")
    for agent in 1:12
        color = A.agent_color(sparse,agent)
        center = A.sx(A.CENTERS[agent],X0)
        A.rect(io,X0+(agent-1)*A.W/12+2,top,A.W/12-4,12;fill=color,radius=2,
            extra="data-actuator=\"$agent\"")
        A.line(io,center,top+16,center,top+25;color,sw=1.5)
        robot_icon(io,robot,prefix,agent,color,center,top+34)
    end
    println(io,"</g>")
end

function panel(io, data, robot, sparse, offset)
    prefix = sparse ? "sparse" : "dense"
    println(io,"<g id=\"$prefix-panel\" data-controller=\"$prefix\" transform=\"translate(0 $(A.fmt(offset)))\">")
    # Reuse the selected variant's sensor and bracket layers verbatim, with
    # panel-specific IDs so the combined SVG never contains duplicate IDs.
    buffer = IOBuffer()
    A.sensor_layer(buffer,data,sparse;x0=X0,y0=Y0)
    A.window_layer(buffer,collect(1:12),sparse;x0=X0,y0=Y0,style=:brackets)
    body = replace(String(take!(buffer)),
        "id=\"sensor-points\"" => "id=\"$prefix-sensor-points\"",
        "id=\"agent-windows\"" => "id=\"$prefix-agent-windows\"")
    println(io,body)
    robot_actuators(io,robot,sparse;top=Y0+A.H+26)
    println(io,"</g>")
end

function main(args=ARGS)
    # Same source-selection options as the asset builder; fixed 12-agent layout.
    if "--help" in args
        println("Usage: julia --startup-file=no --project=. \"Revision/Main Figure/make_robot_export.jl\"\n",
            "  [--output-dir PATH] [--experiment-id ID] [--state-file PATH] [--check-only]\n",
            "Reads robot.svg beside this script. Writes one stacked, transparent SVG and provenance JSON.")
        return
    end
    any(arg -> arg == "--agents",args) && error("This selected export always shows all 12 agents.")
    options = A.parse_arguments(vcat(["--output-dir",joinpath(@__DIR__,"exports"),"--agents","all"],args))
    data = A.load_inputs(options)
    robot = read_robot(ROBOT_PATH)
    robot_bottom = Y0+A.H+26+34+ROBOT_HEIGHT
    panel_height = ceil(robot_bottom+18)
    sparse_offset = panel_height+PANEL_GAP
    height = sparse_offset+panel_height
    options.check && return println("Validated selected GO-GC mask, 12-agent geometry and $(robot.path_count)-path vector robot.")
    mkpath(options.output)
    output = joinpath(options.output,OUTPUT_STEM*".svg")
    A.write_svg(output,PANEL_WIDTH,height,"Dense and sparse controllers with 12 robot agents",
        "Dense above sparse. All twelve observation windows shown with brackets only. Each sensor point represents T/w/u, colored by temperature. No visible panel headings, footers, bottom numbers or agent boxes. The supplied robot represents each agent in the matching purple or magenta shade.") do io
        panel(io,data,robot,false,0.0)
        panel(io,data,robot,true,sparse_offset)
    end
    provenance = Dict(
        "export"=>OUTPUT_STEM, "svg"=>basename(output),
        "panel_order"=>["dense","sparse"], "window_style"=>"brackets", "agents"=>collect(1:12),
        "visualized_channel"=>"temperature", "sensor_point_represents"=>collect(A.CHANNEL_NAMES),
        "robot_file"=>A.relative(ROBOT_PATH), "robot_sha256"=>A.file_hash(ROBOT_PATH),
        "robot_paths_per_agent"=>robot.path_count, "robot_count"=>24,
        "robot_height"=>ROBOT_HEIGHT, "panel_gap"=>PANEL_GAP, "sparse_panel_offset"=>sparse_offset,
        "width"=>PANEL_WIDTH, "height"=>height,
        "state_file"=>A.relative(options.state), "state_sha256"=>A.file_hash(options.state),
        "selection_file"=>A.relative(data.selection_path), "selection_sha256"=>A.file_hash(data.selection_path),
        "candidate_id"=>data.candidate[:candidate_id], "selection_uses_test_data"=>false,
        "active_sensor_locations"=>count(data.mask[1,:,:]), "active_scalar_inputs"=>count(data.mask),
        "agent_colors"=>Dict(prefix=>[A.agent_color(sparse,a) for a in 1:12]
            for (prefix,sparse) in (("dense",false),("sparse",true))),
        "source_hashes"=>Dict(name=>A.file_hash(joinpath(@__DIR__,name))
            for name in ("make_robot_export.jl","make_assets.jl")),
    )
    open(joinpath(options.output,OUTPUT_STEM*".json"),"w") do io
        JSON.print(io,provenance,2)
        println(io)
    end
    println("Wrote $output ($PANEL_WIDTH × $height), 24 inline vector robots.")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    MainFigureRobotExport.main()
end
