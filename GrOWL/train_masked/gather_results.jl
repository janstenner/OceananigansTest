using FileIO
using Statistics
using Printf
using PlotlyJS

const RUN_NUMBERS = 1001:1010
const ROLLING_WINDOW = 50

const MAT_COLOR = "rgb(31, 119, 180)"
const MAT_FILL = "rgba(31, 119, 180, 0.40)"
const MASKED_COLOR = "rgb(214, 39, 40)"
const MASKED_FILL = "rgba(214, 39, 40, 0.40)"

const FIXED_IC_ALGORITHMS = (
    (
        label = "MAT",
        saves_dir = joinpath(@__DIR__, "..", "..", "RB_Joon", "saves"),
        filename = (run_number -> "hook$(run_number).jld2"),
    ),
    (
        label = "masked MAT",
        saves_dir = joinpath(@__DIR__, "saves"),
        filename = (run_number -> "hook$(run_number).jld2"),
    ),
)

const RANDOM_IC_ALGORITHMS = (
    (
        label = "MAT",
        saves_dir = joinpath(@__DIR__, "..", "..", "randomIC", "saves"),
        filename = (run_number -> "hookMAT$(run_number).jld2"),
    ),
    (
        label = "masked MAT",
        saves_dir = joinpath(@__DIR__, "saves"),
        filename = (run_number -> "hook_rIC$(run_number).jld2"),
    ),
)

const COMPARISONS = (
    (
        name = "fixedIC",
        title = "Reward comparison (RB_Joon MAT vs masked MAT)",
        output_filename = "reward_comparison_MAT_vs_maskedMAT_fixedIC.html",
        algorithms = FIXED_IC_ALGORITHMS,
    ),
    (
        name = "randomIC",
        title = "RandomIC reward comparison (MAT vs masked MAT)",
        output_filename = "reward_comparison_MAT_vs_maskedMAT_randomIC.html",
        algorithms = RANDOM_IC_ALGORITHMS,
    ),
)

reward_vector(hook) = Float64.(collect(hook.rewards))

function load_hooks(algorithm_specs)
    hooks = Dict{String, Dict{Int, Any}}(spec.label => Dict{Int, Any}() for spec in algorithm_specs)

    missing_files = String[]
    failed_loads = NamedTuple[]

    for spec in algorithm_specs
        for run_number in RUN_NUMBERS
            path = joinpath(spec.saves_dir, spec.filename(run_number))
            if !isfile(path)
                push!(missing_files, path)
                continue
            end

            try
                hook = FileIO.load(path, "hook")
                hooks[spec.label][run_number] = hook
                var_suffix = replace(spec.label, r"[^A-Za-z0-9]+" => "_")
                @eval global $(Symbol("hook_", var_suffix, "_", run_number)) = $hook
            catch err
                push!(
                    failed_loads,
                    (
                        label = spec.label,
                        run_number = run_number,
                        path = path,
                        err_msg = sprint(showerror, err),
                    ),
                )
            end
        end
    end

    if !isempty(missing_files)
        println("Missing hook files (skipping):")
        println(join(missing_files, "\n"))
        println("Expected files for runs $(first(RUN_NUMBERS)) to $(last(RUN_NUMBERS)).")
    end

    if !isempty(failed_loads)
        println("Could not load some hooks (skipping):")
        for row in failed_loads
            @printf(
                "  [%s run %d] %s\n      %s\n",
                row.label,
                row.run_number,
                row.path,
                row.err_msg,
            )
        end
    end

    for spec in algorithm_specs
        loaded = sort!(collect(keys(hooks[spec.label])))
        println("Loaded $(length(loaded)) $(spec.label) hooks: $(isempty(loaded) ? "none" : join(loaded, ", "))")
    end

    return hooks
end

function reward_by_run(hooks_for_algorithm::AbstractDict)
    Dict(run_number => reward_vector(hook) for (run_number, hook) in hooks_for_algorithm)
end

function summarize_rewards(algorithm::String, run_rewards::AbstractDict)
    if isempty(run_rewards)
        @printf("[%s] no runs available; skipping aggregate statistics.\n", algorithm)
        return nothing
    end

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
        @printf(
            "[%s] need at least %d rewards per run to compute rolling mean; minimum available length is %d. Skipping aggregate statistics.\n",
            algorithm,
            ROLLING_WINDOW,
            min_len,
        )
        return nothing
    end

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
    mean_vec = vec(mean(stacked, dims = 2))
    std_vec = vec(std(stacked, dims = 2; corrected = false))
    return (mean_vec, std_vec, min_len)
end

function style_for_algorithm(algorithm::String)
    if occursin("masked", lowercase(algorithm))
        return (MASKED_COLOR, MASKED_FILL)
    end
    return (MAT_COLOR, MAT_FILL)
end

function add_mean_and_std_traces!(
    traces::Vector{AbstractTrace},
    algorithm::String,
    stats,
)
    if stats === nothing
        return
    end

    mean_vec, std_vec, _ = stats
    steps = 1:length(mean_vec)
    line_color, fill_color = style_for_algorithm(algorithm)

    push!(
        traces,
        scatter(
            x = steps,
            y = mean_vec .+ std_vec,
            mode = "lines",
            line = attr(width = 0),
            name = "$algorithm +1 std",
            showlegend = false,
            hoverinfo = "skip",
        ),
    )
    push!(
        traces,
        scatter(
            x = steps,
            y = mean_vec .- std_vec,
            mode = "lines",
            line = attr(width = 0),
            fill = "tonexty",
            fillcolor = fill_color,
            name = "$algorithm ±1 std",
        ),
    )
    push!(
        traces,
        scatter(
            x = steps,
            y = mean_vec,
            mode = "lines",
            line = attr(color = line_color, width = 2),
            name = "$algorithm mean",
        ),
    )
end

function plot_comparison(stats_by_algorithm::AbstractDict; title::String, output_filename::String)
    traces = AbstractTrace[]

    add_mean_and_std_traces!(traces, "MAT", get(stats_by_algorithm, "MAT", nothing))
    add_mean_and_std_traces!(traces, "masked MAT", get(stats_by_algorithm, "masked MAT", nothing))

    if isempty(traces)
        println("No aggregate traces available to plot for: $title")
        return
    end

    layout = Layout(
        title = title,
        xaxis_title = "Step",
        yaxis_title = "Reward",
        template = "plotly_white",
    )

    fig = plot(traces, layout)
    display(fig)

    output_path = joinpath(@__DIR__, "saves", output_filename)
    try
        savefig(fig, output_path)
        println("Saved comparison plot to: $output_path")
    catch err
        println("Could not save comparison plot to HTML: $(typeof(err))")
    end
end

function print_last_100_ranking(run_rewards_by_algorithm::AbstractDict, algorithm_order)
    ranking = NamedTuple[]
    for algorithm in algorithm_order
        if !haskey(run_rewards_by_algorithm, algorithm)
            continue
        end
        for run_number in sort!(collect(keys(run_rewards_by_algorithm[algorithm])))
            rewards = run_rewards_by_algorithm[algorithm][run_number]
            n = min(100, length(rewards))
            tail_mean = n == 0 ? NaN : mean(@view rewards[(end - n + 1):end])
            push!(ranking, (algorithm = algorithm, run_number = run_number, tail_mean = tail_mean, n = n))
        end
    end

    if isempty(ranking)
        println("\nNo runs available for last-100 ranking.")
        return ranking
    end

    sort!(ranking, by = row -> row.tail_mean, rev = true)

    println("\nRanking by mean reward over the last 100 steps:")
    for (idx, row) in enumerate(ranking)
        @printf(
            "%2d. %-10s run %4d | mean(last %3d) = %12.6f\n",
            idx,
            row.algorithm,
            row.run_number,
            row.n,
            row.tail_mean,
        )
    end

    return ranking
end

function print_best_window_ranking(run_rewards_by_algorithm::AbstractDict, algorithm_order; window::Int = 100)
    if window <= 0
        error("window must be positive, got $window")
    end

    ranking = NamedTuple[]
    for algorithm in algorithm_order
        if !haskey(run_rewards_by_algorithm, algorithm)
            continue
        end

        for run_number in sort!(collect(keys(run_rewards_by_algorithm[algorithm])))
            rewards = run_rewards_by_algorithm[algorithm][run_number]
            n = length(rewards)
            if n < window
                @printf(
                    "[%s run %d] has only %d rewards, fewer than window=%d. Skipping.\n",
                    algorithm,
                    run_number,
                    n,
                    window,
                )
                continue
            end

            best_start = 1
            best_end = window
            window_sum = sum(@view rewards[1:window])
            best_mean = window_sum / window

            for start_idx in 2:(n - window + 1)
                window_sum += rewards[start_idx + window - 1] - rewards[start_idx - 1]
                current_mean = window_sum / window
                if current_mean > best_mean
                    best_mean = current_mean
                    best_start = start_idx
                    best_end = start_idx + window - 1
                end
            end

            push!(
                ranking,
                (
                    algorithm = algorithm,
                    run_number = run_number,
                    best_mean = best_mean,
                    best_start = best_start,
                    best_end = best_end,
                ),
            )
        end
    end

    if isempty(ranking)
        println("\nNo runs available for best-window ranking.")
        return ranking
    end

    sort!(ranking, by = row -> row.best_mean, rev = true)

    println("\nRanking by best $window-step reward window mean:")
    for (idx, row) in enumerate(ranking)
        @printf(
            "%2d. %-10s run %4d | mean(%4d:%4d) = %12.6f\n",
            idx,
            row.algorithm,
            row.run_number,
            row.best_start,
            row.best_end,
            row.best_mean,
        )
    end

    return ranking
end

function run_comparison(comparison)
    println("\n=== $(comparison.name) comparison ===")

    hooks = load_hooks(comparison.algorithms)

    run_rewards_by_algorithm = Dict(
        spec.label => reward_by_run(hooks[spec.label]) for spec in comparison.algorithms
    )

    stats_by_algorithm = Dict(
        spec.label => summarize_rewards(spec.label, run_rewards_by_algorithm[spec.label]) for spec in comparison.algorithms
    )

    plot_comparison(
        stats_by_algorithm;
        title = comparison.title,
        output_filename = comparison.output_filename,
    )

    algorithm_order = Tuple(spec.label for spec in comparison.algorithms)
    print_last_100_ranking(run_rewards_by_algorithm, algorithm_order)
    print_best_window_ranking(run_rewards_by_algorithm, algorithm_order; window = 100)
end

function main()
    for comparison in COMPARISONS
        run_comparison(comparison)
    end
end

main()
