using Flux
using JLD2
using PlotlyJS
using RL
using Statistics

const EXPERT_TRAINING_DIRECTORY = @__DIR__
const REVISION_DIRECTORY = normpath(joinpath(EXPERT_TRAINING_DIRECTORY, ".."))
const PROJECT_ROOT = normpath(joinpath(REVISION_DIRECTORY, ".."))
const RUN_ID = "seed_ce0b5b582dda8eff"
const CHECKPOINT_PATH = joinpath(
    REVISION_DIRECTORY,
    "MAT_IPPO_Comparison",
    "results",
    "runs",
    RUN_ID,
    "fixed",
    "mat.jld2",
)
const VALIDATION_PATH = joinpath(
    REVISION_DIRECTORY,
    "MAT_IPPO_Comparison",
    "results",
    "runs",
    RUN_ID,
    "fixed",
    "mat_validation.jld2",
)
const OUTPUT_DIRECTORY = joinpath(EXPERT_TRAINING_DIRECTORY, "results", "fixed_best_seed_rollout")
const RUNTIME_DIRECTORY = joinpath(EXPERT_TRAINING_DIRECTORY, "runtime", "fixed_best_seed_rollout")
const EPISODE_STEPS = 200
const EXPECTED_VALIDATION_SCORE = -588.662153945814

isfile(CHECKPOINT_PATH) || error("Missing MAT checkpoint: $CHECKPOINT_PATH")
isfile(VALIDATION_PATH) || error("Missing MAT validation result: $VALIDATION_PATH")
mkpath(OUTPUT_DIRECTORY)
mkpath(RUNTIME_DIRECTORY)

checkpoint_metadata = JLD2.jldopen(CHECKPOINT_PATH, "r") do file
    (
        status = string(read(file, "status")),
        protocol = Symbol(read(file, "protocol")),
        algorithm = Symbol(read(file, "algorithm")),
        run_seed = Int(read(file, "run_seed")),
    )
end
validation_metadata = JLD2.jldopen(VALIDATION_PATH, "r") do file
    (
        status = string(read(file, "status")),
        run_id = string(read(file, "run_id")),
        validation_mean = Float64(read(file, "validation_mean")),
    )
end

checkpoint_metadata.status == "complete" || error("Checkpoint is not complete.")
checkpoint_metadata.protocol === :fixed || error("Checkpoint is not Fixed IC.")
checkpoint_metadata.algorithm === :mat || error("Checkpoint is not MAT.")
validation_metadata.status == "complete" || error("Validation is not complete.")
validation_metadata.run_id == RUN_ID || error("Validation belongs to the wrong run.")
isapprox(validation_metadata.validation_mean, EXPECTED_VALIDATION_SCORE; atol = 1e-12) ||
    error("Unexpected stored validation score $(validation_metadata.validation_mean).")

ENV["REVISION_RUN_SEED"] = string(checkpoint_metadata.run_seed)
ENV["REVISION_RUN_DIRECTORY"] = RUNTIME_DIRECTORY
include(joinpath(REVISION_DIRECTORY, "Run_Files", "FixedIC_MAT.jl"))

function reset_fixed_episode!()
    initial_state = Base.invokelatest(generate_random_init)
    env.y0 = initial_state
    Base.invokelatest(reset!, env)
    return nothing
end

function run_rollout(; exploration::Bool)
    global agent = JLD2.load(CHECKPOINT_PATH, "agent")
    Flux.testmode!(agent.policy)
    reset_fixed_episode!()

    rewards = Vector{Float64}(undef, EPISODE_STEPS)
    for step in 1:EPISODE_STEPS
        action = exploration ? agent.policy(env) : RL.prob(agent.policy, env).μ
        env(action)
        rewards[step] = mean(Float64.(reward(env)))
    end
    return rewards
end

deterministic_rewards = run_rollout(; exploration = false)
exploratory_rewards = run_rollout(; exploration = true)
deterministic_score = sum(deterministic_rewards)
exploratory_score = sum(exploratory_rewards)

isapprox(deterministic_score, validation_metadata.validation_mean; atol = 1e-6) || error(
    "Deterministic rollout score $deterministic_score does not reproduce stored validation " *
    "score $(validation_metadata.validation_mean).",
)

steps = collect(1:EPISODE_STEPS)
plot_handle = Plot(
    [
        scatter(
            x = steps,
            y = deterministic_rewards,
            mode = "lines",
            name = "Exploration off (mean action)",
            line = attr(color = "#2166AC", width = 3),
        ),
        scatter(
            x = steps,
            y = exploratory_rewards,
            mode = "lines",
            name = "Exploration on (sampled action)",
            line = attr(color = "#B2182B", width = 2.5),
        ),
    ],
    Layout(
        template = "plotly_white",
        title = attr(
            text = "Best Fixed-IC MAT agent — one rollout per exploration mode",
            x = 0.5,
            xanchor = "center",
        ),
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        width = 1000,
        height = 600,
        margin = attr(l = 95, r = 35, t = 80, b = 80),
        font = attr(family = "Arial, sans-serif", size = 15, color = "#303030"),
        xaxis = attr(
            title = "Control step",
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        yaxis = attr(
            title = "Mean environment reward (higher is better)",
            showline = true,
            mirror = true,
            linecolor = "#3A3A3A",
            ticks = "outside",
            gridcolor = "#E6E6E6",
            zeroline = false,
        ),
        legend = attr(
            x = 0.985,
            y = 0.02,
            xanchor = "right",
            yanchor = "bottom",
            bgcolor = "rgba(255, 255, 255, 0.92)",
            bordercolor = "#CFCFCF",
            borderwidth = 1,
        ),
        annotations = [
            attr(
                x = 0.015,
                y = 0.02,
                xref = "paper",
                yref = "paper",
                xanchor = "left",
                yanchor = "bottom",
                showarrow = false,
                bgcolor = "rgba(255, 255, 255, 0.85)",
                bordercolor = "#CFCFCF",
                borderwidth = 1,
                text = "Returns: off=$(round(deterministic_score; digits = 3)), " *
                       "on=$(round(exploratory_score; digits = 3))",
            ),
        ],
        hovermode = "x unified",
    ),
)

svg_path = joinpath(OUTPUT_DIRECTORY, "reward_curves.svg")
png_path = joinpath(OUTPUT_DIRECTORY, "reward_curves.png")
PlotlyJS.savefig(plot_handle, svg_path; width = 1000, height = 600)
PlotlyJS.savefig(plot_handle, png_path; width = 1000, height = 600)

println("Checkpoint: $CHECKPOINT_PATH")
println("Stored validation score: $(validation_metadata.validation_mean)")
println("Exploration off score: $deterministic_score")
println("Exploration on score:  $exploratory_score")
println("Reward plot (SVG): $svg_path")
println("Reward plot (PNG): $png_path")
