include(joinpath(@__DIR__, "randomIC_MAT.jl"))

using FileIO
using Statistics
using Printf
using PlotlyJS

const RUN_NUMBERS = 1001:1010
const ALGORITHMS = ("MAT", "PPO")
const SAVES_DIR = joinpath(@__DIR__, "saves")
const ROLLING_WINDOW = 50

const HOOKS = Dict(
    "MAT" => Dict{Int, Any}(),
    "PPO" => Dict{Int, Any}(),
)

function load_hooks!()
    missing_files = String[]
    for algorithm in ALGORITHMS
        for run_number in RUN_NUMBERS
            path = joinpath(SAVES_DIR, "hook$(algorithm)$(run_number).jld2")
            if !isfile(path)
                push!(missing_files, path)
                continue
            end

            hook = FileIO.load(path, "hook")
            HOOKS[algorithm][run_number] = hook
            @eval global $(Symbol("hook$(algorithm)$(run_number)")) = $hook
        end
    end

    if !isempty(missing_files)
        error(
            "Missing hook files:\n" * join(missing_files, "\n") *
            "\nExpected files for runs $(first(RUN_NUMBERS)) to $(last(RUN_NUMBERS))."
        )
    end
end

reward_vector(hook) = Float64.(collect(hook.rewards))

function reward_by_run(algorithm::String)
    Dict(run_number => reward_vector(HOOKS[algorithm][run_number]) for run_number in RUN_NUMBERS)
end

function summarize_rewards(algorithm::String, run_rewards::Dict{Int, Vector{Float64}})
    run_ids = sort!(collect(keys(run_rewards)))
    lengths = [length(run_rewards[run_id]) for run_id in run_ids]
    min_len = minimum(lengths)
    max_len = maximum(lengths)

    if min_len != max_len
        @printf(
            "[%s] reward lengths differ across runs; truncating all to %d steps for aggregate stats.\n",
            algorithm,
            min_len,
        )
    end

    if min_len < ROLLING_WINDOW
        error(
            "[$algorithm] need at least $ROLLING_WINDOW rewards per run to compute rolling mean; " *
            "minimum available length is $min_len."
        )
    end

    # Keep run_rewards untouched: build rolling means from truncated views.
    rolling_len = min_len - ROLLING_WINDOW + 1
    rolling_by_run = Vector{Vector{Float64}}(undef, length(run_ids))
    for (idx, run_id) in enumerate(run_ids)
        rewards = run_rewards[run_id]
        truncated = @view rewards[1:min_len]
        rolling = Vector{Float64}(undef, rolling_len)
        window_sum = sum(@view truncated[1:ROLLING_WINDOW])
        rolling[1] = window_sum / ROLLING_WINDOW
        for i in 2:rolling_len
            window_sum += truncated[i + ROLLING_WINDOW - 1] - truncated[i - 1]
            rolling[i] = window_sum / ROLLING_WINDOW
        end
        rolling_by_run[idx] = rolling
    end

    stacked = hcat(rolling_by_run...)
    mean_vec = vec(mean(stacked, dims=2))
    std_vec = vec(std(stacked, dims=2; corrected=false))
    return mean_vec, std_vec, min_len
end

function plot_comparison(mat_mean, mat_std, ppo_mean, ppo_std)
    mat_steps = 1:length(mat_mean)
    ppo_steps = 1:length(ppo_mean)

    traces = [
        scatter(
            x=mat_steps,
            y=mat_mean .+ mat_std,
            mode="lines",
            line=attr(width=0),
            name="MAT +1 std",
            showlegend=false,
            hoverinfo="skip",
        ),
        scatter(
            x=mat_steps,
            y=mat_mean .- mat_std,
            mode="lines",
            line=attr(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.40)",
            name="MAT ±1 std",
        ),
        scatter(
            x=ppo_steps,
            y=ppo_mean .+ ppo_std,
            mode="lines",
            line=attr(width=0),
            name="PPO +1 std",
            showlegend=false,
            hoverinfo="skip",
        ),
        scatter(
            x=ppo_steps,
            y=ppo_mean .- ppo_std,
            mode="lines",
            line=attr(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.40)",
            name="PPO ±1 std",
        ),
        scatter(
            x=ppo_steps,
            y=ppo_mean,
            mode="lines",
            line=attr(color="rgb(255, 127, 14)", width=2),
            name="PPO mean",
        ),
        scatter(
            x=mat_steps,
            y=mat_mean,
            mode="lines",
            line=attr(color="rgb(31, 119, 180)", width=2),
            name="MAT mean",
        ),
    ]

    layout = Layout(
        title="RandomIC reward comparison (MAT vs PPO)",
        xaxis_title="Step",
        yaxis_title="Reward",
        template="plotly_white",
    )

    fig = plot(traces, layout)
    display(fig)

    output_path = joinpath(@__DIR__, "saves", "reward_comparison_MAT_vs_PPO.html")
    try
        savefig(fig, output_path)
        println("Saved comparison plot to: $output_path")
    catch err
        println("Could not save comparison plot to HTML: $(typeof(err))")
    end
end

function print_last_100_ranking(run_rewards_by_algorithm::Dict{String, Dict{Int, Vector{Float64}}})
    ranking = NamedTuple[]
    for algorithm in ALGORITHMS
        for run_number in RUN_NUMBERS
            rewards = run_rewards_by_algorithm[algorithm][run_number]
            n = min(100, length(rewards))
            tail_mean = n == 0 ? NaN : mean(@view rewards[(end - n + 1):end])
            push!(ranking, (algorithm=algorithm, run_number=run_number, tail_mean=tail_mean, n=n))
        end
    end

    sort!(ranking, by=row -> row.tail_mean, rev=true)

    println("\nRanking by mean reward over the last 100 steps:")
    for (idx, row) in enumerate(ranking)
        @printf(
            "%2d. %-3s run %4d | mean(last %3d) = %12.6f\n",
            idx, row.algorithm, row.run_number, row.n, row.tail_mean
        )
    end
end

function main()
    load_hooks!()

    run_rewards_by_algorithm = Dict(
        "MAT" => reward_by_run("MAT"),
        "PPO" => reward_by_run("PPO"),
    )

    mat_mean, mat_std, _ = summarize_rewards("MAT", run_rewards_by_algorithm["MAT"])
    ppo_mean, ppo_std, _ = summarize_rewards("PPO", run_rewards_by_algorithm["PPO"])

    plot_comparison(mat_mean, mat_std, ppo_mean, ppo_std)
    print_last_100_ranking(run_rewards_by_algorithm)
end

main()
