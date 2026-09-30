# Manuscript Figure 2 (RBC control setup) and Figure 3(b) (sensor window).
#
# Run from the repository root:
#   julia --startup-file=no --project=. "Revision/Main Figure/make_setup_figures.jl"
#
# (a) The saved two-plume state RBmodel300.jld2, which is the Fixed-IC initial condition.
# (b) The state after the deterministic 200-step Fixed-IC expert test episode started from (a).
# (c) The expert actions of that last step and the bottom temperature they impose through the
#     run file's own bottom_T profile.
# Window: the temperature channel plus positional encoding of the 47-column window around
#     agent 6 (the middle agent sketched in Figure 3a) at the state of (a).
# The rollout is checked against the stored Revision/Baselines Fixed-IC expert actions. Every
# panel is sized for its LaTeX inclusion width, giving about 7.5 pt text like the other figures.

using Dates
using JSON
using SHA

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTPUT_DIR = joinpath(@__DIR__, "exports", "setup")
const RUN_FILE = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "FixedIC_MAT.jl")
const DISTILLATION_ROOT = joinpath(PROJECT_ROOT, "Revision", "Expert_Apprentice_Distillation")
const EXPERT_PATH = joinpath(DISTILLATION_ROOT, "experts", "fixed", "agent.jld2")
const BASELINE_PATH = joinpath(PROJECT_ROOT, "Revision", "Baselines", "results", "fixed", "expert.jld2")
const STATE_PATH = joinpath(PROJECT_ROOT, "RBmodel300.jld2")
const STEPS = 200
const WINDOW_COLUMNS = 47
const WINDOW_AGENT = 6

# Temperature colors of the original setup figures (RBC_analyse/sensor_plot.jl).
const TEMPERATURE_STOPS = [
    (0.0, "rgb(41, 100, 189)"), (0.19, "rgb(33, 166, 196)"), (0.33, "rgb(246, 230, 170)"),
    (0.5, "rgb(220, 140, 49)"), (0.62, "rgb(227, 67, 18)"), (1.0, "rgb(252, 96, 255)"),
]
const TEMPERATURE_RANGE = (1.0, 2.5)
# Temperature plus positional encoding spans about 0.2 to 2.95. Spreading it over 0 to 4 uses the
# colors like the temperature fields do, reserving magenta for the extremes.
const WINDOW_RANGE = (0.0, 4.0)
# Figure 2(c): heating (above T_b) and cooling (below T_b) colors, taken from the temperature stops.
const WARM = (229, 70, 42)
const COOL = (34, 158, 195)
const INK = "#303030"
const FRAME = "#3A3A3A"
const GRID = "#E6E6E6"

# Panel sizes in px and the LaTeX widths they are drawn for (\columnwidth = 433.62 pt).
const PANEL_WIDTH, PANEL_HEIGHT = 560, 420        # Figure 2 panels at 0.32\columnwidth
const WINDOW_WIDTH, WINDOW_HEIGHT = 840, 470      # Figure 3(b) at 0.48\columnwidth
const TICK_SIZE = 30
const TITLE_SIZE = 32

file_sha256(path) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

# ---------------------------------------------------------------------------------------------
# Deterministic Fixed-IC expert rollout, set up exactly like Revision/Baselines/run_baseline.jl.

ENV["REVISION_RUN_SEED"] = "600600"
ENV["REVISION_RUN_DIRECTORY"] = mktempdir(; prefix = "setup-figures-")
ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
include(RUN_FILE)
include(joinpath(DISTILLATION_ROOT, "DistillationCorpus.jl"))
expert_metadata = load_distillation_expert!(:fixed; explicit_path = EXPERT_PATH)

PlotlyJS.PlotlyKaleido.kill_kaleido()
PlotlyJS.PlotlyKaleido.start(plotlyjs = PlotlyJS._js_path, mathjax = false)

generate_random_init()
RL.reset!(env)
initial_state = copy(env.y)
stored_temperature = Float64.(values["b/data"][4:Nx+3, 1, 4:Nz+3])
maximum(abs.(initial_state[1, :, :] .- stored_temperature)) < 1e-5 ||
    error("The reset state differs from RBmodel300.jld2.")

rollout_actions = Matrix{Float32}(undef, STEPS, actuators)
rollout_nusselt = Vector{Float64}(undef, STEPS)
for step in 1:STEPS
    action = RL.prob(agent.policy, env).μ
    hasproperty(agent.policy, :clip1) && agent.policy.clip1 && clamp!(action, -1.0f0, 1.0f0)
    rollout_actions[step, :] .= vec(Float32.(Array(action)))
    env(action)
    rollout_nusselt[step] = state_Nu(env)
end
controlled_state = copy(env.y)
final_actions = Float64.(rollout_actions[end, :])
maximum(abs.(Float64.(actions) .- final_actions)) < 1e-6 ||
    error("The boundary function does not hold the last applied actions.")

baseline = only(JLD2.load(BASELINE_PATH, "episodes"))
action_deviation = maximum(abs.(rollout_actions .- baseline.actions))
nusselt_deviation = abs(mean(rollout_nusselt) - baseline.mean_state_nusselt)
println("Rollout vs. stored baseline: max |Δa| = $action_deviation, |Δ mean state_Nu| = $nusselt_deviation")
nusselt_deviation < 1e-3 || error("The rollout does not reproduce the stored Fixed-IC expert episode.")

x_boundary = collect(range(0, Lx - 1e-7; length = 2401))
boundary_temperature = [bottom_T(value, 0.0) for value in x_boundary]

# Temperature plus positional encoding at the sensors, as in the run file's featurize.
function sensor_window(state, agent_index)
    sensordata = state[:, sensor_positions[1], sensor_positions[2]]
    for column in 1:sensors[1]
        sensordata[1, column, :] .+= sin(2π * column / sensors[1])
    end
    half = WINDOW_COLUMNS ÷ 2
    center = actuators_to_sensors[agent_index]
    columns = [mod1(center + offset, sensors[1]) for offset in -half:half]
    return sensordata[1, columns, :], collect(-half:half)
end
window_values, window_offsets = sensor_window(initial_state, WINDOW_AGENT)

# ---------------------------------------------------------------------------------------------
# Plot helpers.

rgba_string(color, alpha) = "rgba($(color[1]), $(color[2]), $(color[3]), $alpha)"
rgb_string(color) = "rgb($(color[1]), $(color[2]), $(color[3]))"
plotly_colorscale(stops) = [[position, color] for (position, color) in stops]

function paper_axis(title; kwargs...)
    return attr(;
        title = attr(text = title, standoff = 8, font = attr(size = TITLE_SIZE, color = INK)),
        tickfont = attr(size = TICK_SIZE, color = INK),
        showline = true, mirror = true, linecolor = FRAME, linewidth = 1.5,
        ticks = "outside", ticklen = 6, tickcolor = FRAME,
        showgrid = false, zeroline = false,
        kwargs...,
    )
end

function base_layout(width, height; margin, kwargs...)
    return Layout(;
        template = "plotly_white", width, height,
        paper_bgcolor = "white", plot_bgcolor = "white",
        font = attr(family = "Arial, sans-serif", size = TICK_SIZE, color = INK),
        margin, showlegend = false,
        kwargs...,
    )
end

function temperature_colorbar(title, tickvals)
    return attr(
        title = attr(text = title, side = "top", font = attr(size = TITLE_SIZE, color = INK)),
        tickvals = tickvals, tickfont = attr(size = TICK_SIZE, color = INK),
        thickness = 20, len = 1.0, lenmode = "fraction", y = 0.5, yanchor = "middle",
        outlinewidth = 1, outlinecolor = FRAME, ticks = "outside", ticklen = 4,
    )
end

function save_figure(figure, stem, width, height)
    mkpath(OUTPUT_DIR)
    paths = String[]
    for extension in ("svg", "pdf")
        path = joinpath(OUTPUT_DIR, "$stem.$extension")
        PlotlyJS.savefig(figure, path; width, height)
        push!(paths, path)
    end
    println("Wrote $(join(paths, ", "))")
    return paths
end

function field_panel(temperature, stem)
    x = ((1:Nx) .- 0.5) .* (Lx / Nx)
    y = ((1:Nz) .- 0.5) .* (Lz / Nz)
    heat = heatmap(
        x = x, y = y, z = permutedims(temperature),
        colorscale = plotly_colorscale(TEMPERATURE_STOPS),
        zmin = TEMPERATURE_RANGE[1], zmax = TEMPERATURE_RANGE[2], zsmooth = "best",
        colorbar = temperature_colorbar("<i>T</i>", [1.0, 1.5, 2.0, 2.5]),
        hovertemplate = "x=%{x:.2f}<br>y=%{y:.2f}<br>T=%{z:.3f}<extra></extra>",
    )
    layout = base_layout(PANEL_WIDTH, PANEL_HEIGHT;
        margin = attr(l = 78, r = 12, t = 28, b = 72),
        xaxis = paper_axis("<i>x</i>"; range = [0, Lx], tickvals = [0, 2, 4, 6]),
        yaxis = paper_axis("<i>y</i>"; range = [0, Lz], tickvals = [0, 0.5, 1, 1.5, 2]),
    )
    return save_figure(plot(heat, layout), stem, PANEL_WIDTH, PANEL_HEIGHT)
end

function boundary_control_panel(x, temperature, actions, stem)
    reference = fill(2.0, length(x))
    segment = Lx / actuators
    centers = ((1:actuators) .- 0.5) .* segment
    # The run file keeps the policy means unclipped (clip1 = false); bottom_T normalizes them.
    action_range = [min(minimum(actions), -1.0) - 0.3, max(maximum(actions), 1.0) + 0.3]
    action_ticks = [tick for tick in -3:3 if action_range[1] <= tick <= action_range[2]]
    temperature_range = [minimum(temperature) - 0.15, maximum(temperature) + 0.15]
    figure = make_subplots(rows = 2, cols = 1, shared_xaxes = true,
        vertical_spacing = 0.07, row_heights = [0.56, 0.44])
    # Heating (above T_b) and cooling (below T_b) relative to the mean bottom temperature.
    for (bound, color) in ((max.(temperature, 2.0), WARM), (min.(temperature, 2.0), COOL))
        add_trace!(figure, scatter(x = x, y = reference, mode = "lines", line = attr(width = 0),
            hoverinfo = "skip"); row = 1, col = 1)
        add_trace!(figure, scatter(x = x, y = bound, mode = "lines", line = attr(width = 0),
            fill = "tonexty", fillcolor = rgba_string(color, 0.22), hoverinfo = "skip"); row = 1, col = 1)
    end
    add_trace!(figure, scatter(x = x, y = reference, mode = "lines",
        line = attr(color = "#8A8A8A", width = 1.5, dash = "dash"), hoverinfo = "skip"); row = 1, col = 1)
    add_trace!(figure, scatter(x = x, y = temperature, mode = "lines",
        line = attr(color = INK, width = 3),
        hovertemplate = "x=%{x:.2f}<br>T=%{y:.3f}<extra></extra>"); row = 1, col = 1)
    add_trace!(figure, bar(x = centers, y = actions, width = 0.62 * segment,
        marker = attr(color = [rgb_string(value >= 0 ? WARM : COOL) for value in actions], line = attr(width = 0)),
        hovertemplate = "agent %{customdata}<br>a=%{y:.3f}<extra></extra>",
        customdata = collect(1:actuators)); row = 2, col = 1)
    shapes = Any[]
    for boundary in segment .* (1:actuators-1), axis_name in ("x", "x2")
        yref = axis_name == "x" ? "y domain" : "y2 domain"
        push!(shapes, attr(type = "line", xref = axis_name, yref = yref, x0 = boundary, x1 = boundary,
            y0 = 0, y1 = 1, line = attr(color = GRID, width = 1.5), layer = "below"))
    end
    push!(shapes, attr(type = "line", xref = "x2 domain", yref = "y2", x0 = 0, x1 = 1, y0 = 0, y1 = 0,
        line = attr(color = FRAME, width = 1.2)))
    layout = base_layout(PANEL_WIDTH, PANEL_HEIGHT;
        margin = attr(l = 96, r = 12, t = 16, b = 72), shapes = shapes, bargap = 0,
        xaxis = paper_axis(""; range = [0, Lx], tickvals = [0, 2, 4, 6], showticklabels = false),
        yaxis = paper_axis("<i>T</i>(<i>x</i>, 0)"; range = temperature_range, tickvals = [1.5, 2.0, 2.5]),
        xaxis2 = paper_axis("<i>x</i>"; range = [0, Lx], tickvals = [0, 2, 4, 6]),
        yaxis2 = paper_axis("<i>a<sub>i</sub></i>"; range = action_range, tickvals = action_ticks),
    )
    # Keep the subplot domains and axis links created by make_subplots.
    fields = layout.fields
    for key in (:xaxis, :yaxis, :xaxis2, :yaxis2)
        existing = get(figure.plot.layout.fields, key, Dict{Any, Any}())
        merged = Dict{Symbol, Any}(Symbol(name) => value for (name, value) in existing)
        update = fields[key] isa AbstractDict ? fields[key] : fields[key].fields
        merge!(merged, Dict{Symbol, Any}(Symbol(name) => value for (name, value) in update))
        fields[key] = attr(; merged...)
    end
    relayout!(figure, fields)
    return save_figure(figure, stem, PANEL_WIDTH, PANEL_HEIGHT)
end

function window_panel(values, offsets, stem)
    low, high = WINDOW_RANGE
    heat = heatmap(
        x = offsets, y = collect(1:sensors[2]), z = permutedims(values),
        colorscale = plotly_colorscale(TEMPERATURE_STOPS), zmin = low, zmax = high,
        colorbar = temperature_colorbar("<i>T</i> + PE", [0, 1, 2, 3, 4]),
        hovertemplate = "offset=%{x}<br>row=%{y}<br>T+PE=%{z:.3f}<extra></extra>",
    )
    layout = base_layout(WINDOW_WIDTH, WINDOW_HEIGHT;
        margin = attr(l = 92, r = 12, t = 16, b = 76),
        xaxis = paper_axis("Column offset from agent center"; range = [offsets[1] - 0.5, offsets[end] + 0.5],
            tickvals = [-20, -10, 0, 10, 20]),
        yaxis = paper_axis("Sensor row"; range = [0.5, sensors[2] + 0.5], tickvals = collect(1:sensors[2])),
    )
    return save_figure(plot(heat, layout), stem, WINDOW_WIDTH, WINDOW_HEIGHT)
end

# ---------------------------------------------------------------------------------------------

outputs = vcat(
    field_panel(initial_state[1, :, :], "figure_2a_uncontrolled"),
    field_panel(controlled_state[1, :, :], "figure_2b_controlled"),
    boundary_control_panel(x_boundary, boundary_temperature, final_actions, "figure_2c_boundary_control"),
    window_panel(window_values, window_offsets, "figure_3b_sensor_window"),
)

provenance_path = joinpath(OUTPUT_DIR, "setup_figures_provenance.json")
open(provenance_path, "w") do io
    JSON.print(io, Dict(
        "created_at" => string(Dates.now(Dates.UTC)),
        "script_sha256" => file_sha256(@__FILE__),
        "run_file" => relpath(RUN_FILE, PROJECT_ROOT), "run_file_sha256" => file_sha256(RUN_FILE),
        "state_file" => relpath(STATE_PATH, PROJECT_ROOT), "state_sha256" => file_sha256(STATE_PATH),
        "expert" => relpath(EXPERT_PATH, PROJECT_ROOT), "expert_sha256" => file_sha256(EXPERT_PATH),
        "baseline" => relpath(BASELINE_PATH, PROJECT_ROOT),
        "steps" => STEPS,
        "max_action_deviation_from_baseline" => action_deviation,
        "mean_state_nusselt" => mean(rollout_nusselt),
        "baseline_mean_state_nusselt" => baseline.mean_state_nusselt,
        "final_state_nusselt" => rollout_nusselt[end],
        "final_actions" => final_actions,
        "window_agent" => WINDOW_AGENT, "window_columns" => WINDOW_COLUMNS,
        "outputs" => [relpath(path, PROJECT_ROOT) for path in outputs],
    ), 2)
end
println("Wrote $provenance_path")
