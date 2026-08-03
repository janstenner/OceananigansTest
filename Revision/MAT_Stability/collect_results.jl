ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Dates
using JLD2
using Plots
using Printf
using Statistics

include(joinpath(@__DIR__, "MATStabilityExperiment.jl"))
using .MATStabilityExperiment

const CONFIG_NAMES = Tuple(config.name for config in MAT_CONFIGS)
const PROTOCOLS = (:fixed, :varying)
const FINAL_WINDOW = 100
const CONFIG_DISPLAY_NAMES = Dict(
    :python_like => "non-modified",
    :modified_half => "modified 1",
    :modified_full => "modified 2",
)
const CONFIG_COLORS = Dict(
    :python_like => "#4C78A8",
    :modified_half => "#F58518",
    :modified_full => "#54A24B",
)
const PROTOCOL_DISPLAY_NAMES = Dict(:fixed => "Fixed IC", :varying => "Varying IC")

function usage(io::IO = stdout)
    println(
        io,
        """
        Usage:
          julia --project=. Revision/MAT_Stability/collect_results.jl [options]

        Options:
          --results-dir DIR  Result root (default: Revision/MAT_Stability/results).
          --output-dir DIR   Collector output directory (default: <results>/collected).
          --help             Show this message.

        Missing protocols, replicates, and configurations are ignored. Every
        complete, valid result file that is present is collected.
        """,
    )
end

function parse_arguments(arguments)
    options = Dict{String, Any}(
        "results_dir" => joinpath(@__DIR__, "results"),
        "output_dir" => nothing,
    )

    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument in ("--results-dir", "--output-dir")
            index == length(arguments) && error("Missing value after $argument.")
            options[replace(argument[3:end], "-" => "_")] = arguments[index + 1]
            index += 1
        else
            error("Unknown argument '$argument'. Use --help for usage.")
        end
        index += 1
    end

    options["results_dir"] = abspath(options["results_dir"])
    if isnothing(options["output_dir"])
        options["output_dir"] = joinpath(options["results_dir"], "collected")
    else
        options["output_dir"] = abspath(options["output_dir"])
    end
    return options
end

function result_path(results_directory, protocol, replicate, config_name)
    return joinpath(
        results_directory,
        string(protocol),
        @sprintf("replicate_%02d", replicate),
        string(config_name) * ".jld2",
    )
end

function load_record(path)
    return JLD2.jldopen(path, "r") do file
        status = read(file, "status")
        status == "complete" || return (
            status = status,
            error_message = haskey(file, "error_message") ?
                            read(file, "error_message") : "unknown failure",
        )
        (
            status = status,
            protocol = Symbol(read(file, "protocol")),
            replicate = Int(read(file, "replicate")),
            config_name = Symbol(read(file, "config_name")),
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
            episode_target = Int(read(file, "episode_target")),
            episodes_completed = Int(read(file, "episodes_completed")),
            elapsed_seconds = Float64(read(file, "elapsed_seconds")),
            shared_initial_hash = read(file, "shared_initial_hash"),
            full_initial_hash = read(file, "full_initial_hash"),
            policy_rng_probe = read(file, "policy_rng_probe"),
            initial_condition_probe = read(file, "initial_condition_probe"),
            initial_condition_trace = read(file, "initial_condition_trace"),
            rewards = Float64.(read(file, "rewards")),
            rewards_all_timesteps = Float64.(read(file, "rewards_all_timesteps")),
            errored_episodes = Int.(read(file, "errored_episodes")),
            best_reward = Float64(read(file, "best_reward")),
            best_episode = Int(read(file, "best_episode")),
        )
    end
end

function validate_record(record, protocol, replicate, config_name)
    record.protocol == protocol || error("Protocol mismatch for $protocol/$replicate/$config_name.")
    record.replicate == replicate || error("Replicate mismatch for $protocol/$replicate/$config_name.")
    record.config_name == config_name || error("Config mismatch for $protocol/$replicate/$config_name.")
    expected_episodes = MATStabilityExperiment.DEFAULT_EPISODES[protocol]
    record.episode_target == expected_episodes || error(
        "$protocol/$replicate/$config_name targeted $(record.episode_target), " *
        "expected $expected_episodes episodes.",
    )
    record.episodes_completed == expected_episodes || error(
        "$protocol/$replicate/$config_name completed $(record.episodes_completed), " *
        "expected $expected_episodes episodes.",
    )
    length(record.rewards) == expected_episodes || error(
        "$protocol/$replicate/$config_name contains $(length(record.rewards)) rewards.",
    )
    return
end

function validate_pairing(records, protocol, replicate)
    available = [
        records[(protocol, replicate, config)]
        for config in CONFIG_NAMES
        if haskey(records, (protocol, replicate, config))
    ]
    isempty(available) && return

    reference = first(available)
    for record in Iterators.drop(available, 1)
        record.run_seed == reference.run_seed || error(
            "Run seeds are not paired for $protocol replicate $replicate.",
        )
        record.ic_seed == reference.ic_seed || error(
            "IC seeds are not paired for $protocol replicate $replicate.",
        )
        record.shared_initial_hash == reference.shared_initial_hash || error(
            "Shared initial weights differ for $protocol replicate $replicate.",
        )
        record.policy_rng_probe == reference.policy_rng_probe || error(
            "Policy RNG states differ for $protocol replicate $replicate.",
        )
        record.initial_condition_probe == reference.initial_condition_probe || error(
            "Initial-condition probes differ for $protocol replicate $replicate.",
        )
        if protocol === :varying
            record.initial_condition_trace == reference.initial_condition_trace || error(
                "Varying-IC sequences differ for replicate $replicate.",
            )
        end
    end

    half_key = (protocol, replicate, :modified_half)
    full_key = (protocol, replicate, :modified_full)
    if haskey(records, half_key) && haskey(records, full_key)
        records[half_key].full_initial_hash == records[full_key].full_initial_hash || error(
            "modified_half and modified_full initial networks differ for " *
            "$protocol replicate $replicate.",
        )
    end
    return
end

function collect_records(results_directory)
    records = Dict{Tuple{Symbol, Int, Symbol}, Any}()
    problems = String[]

    for protocol in PROTOCOLS, replicate in 1:5, config_name in CONFIG_NAMES
        path = result_path(results_directory, protocol, replicate, config_name)
        isfile(path) || continue

        record = try
            load_record(path)
        catch error_value
            push!(problems, "unreadable: $path ($(sprint(showerror, error_value)))")
            continue
        end
        if record.status != "complete"
            push!(problems, "failed: $path ($(record.error_message))")
            continue
        end

        try
            validate_record(record, protocol, replicate, config_name)
            records[(protocol, replicate, config_name)] = record
        catch error_value
            push!(problems, "invalid: $path ($(sprint(showerror, error_value)))")
        end
    end

    if !isempty(problems)
        println(stderr, join(problems, "\n"))
        println(stderr, "Ignored $(length(problems)) present but unusable result file(s).")
    end

    for protocol in PROTOCOLS, replicate in 1:5
        validate_pairing(records, protocol, replicate)
    end
    return records, problems
end

function metric_rows(records)
    rows = NamedTuple[]
    for protocol in PROTOCOLS, replicate in 1:5, config_name in CONFIG_NAMES
        key = (protocol, replicate, config_name)
        haskey(records, key) || continue
        record = records[key]
        window = min(FINAL_WINDOW, length(record.rewards))
        final_rewards = @view record.rewards[end-window+1:end]
        push!(
            rows,
            (
                protocol = protocol,
                replicate = replicate,
                config = config_name,
                run_seed = record.run_seed,
                ic_seed = record.ic_seed,
                episodes = record.episodes_completed,
                final_window = window,
                final_reward_mean = mean(final_rewards),
                final_reward_std = std(final_rewards; corrected = false),
                last_reward = last(record.rewards),
                best_reward = record.best_reward,
                best_episode = record.best_episode,
                errored_episode_count = length(record.errored_episodes),
                elapsed_seconds = record.elapsed_seconds,
            ),
        )
    end
    return rows
end

function write_csv(path, rows)
    open(path, "w") do io
        println(
            io,
            "protocol,replicate,config,run_seed,ic_seed,episodes,final_window," *
            "final_reward_mean,final_reward_std,last_reward,best_reward,best_episode," *
            "errored_episode_count,elapsed_seconds",
        )
        for row in rows
            println(
                io,
                join(
                    (
                        row.protocol,
                        row.replicate,
                        row.config,
                        row.run_seed,
                        row.ic_seed,
                        row.episodes,
                        row.final_window,
                        row.final_reward_mean,
                        row.final_reward_std,
                        row.last_reward,
                        row.best_reward,
                        row.best_episode,
                        row.errored_episode_count,
                        row.elapsed_seconds,
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function plot_learning_curves(records, output_directory)
    outputs = String[]

    for protocol in PROTOCOLS
        plot_handle = plot(
            title = "MAT stability - $(PROTOCOL_DISPLAY_NAMES[protocol])",
            xlabel = "Episode",
            ylabel = "Episode reward",
            legend = :bottomright,
            framestyle = :box,
            gridalpha = 0.18,
            size = (850, 520),
            legend_background_color = :transparent,
            legend_foreground_color = :transparent,
        )
        plotted = false
        for config_name in CONFIG_NAMES
            series = [
                records[(protocol, replicate, config_name)].rewards
                for replicate in 1:5
                if haskey(records, (protocol, replicate, config_name))
            ]
            isempty(series) && continue
            episode_count = minimum(length, series)
            episode_count == 0 && continue
            values = reduce(hcat, [run[1:episode_count] for run in series])
            means = vec(mean(values; dims = 2))
            deviations = vec(std(values; dims = 2, corrected = false))
            plot!(
                plot_handle,
                1:episode_count,
                means;
                ribbon = deviations,
                label = "$(CONFIG_DISPLAY_NAMES[config_name]) (n=$(length(series)))",
                color = CONFIG_COLORS[config_name],
                fillalpha = 0.18,
                linewidth = 2.5,
            )
            plotted = true
        end
        plotted || continue
        output = joinpath(output_directory, "learning_curves_$(protocol).svg")
        savefig(plot_handle, output)
        push!(outputs, output)
    end
    return outputs
end

function plot_final_performance(rows, output_directory)
    outputs = String[]
    for protocol in PROTOCOLS
        protocol_rows = filter(row -> row.protocol === protocol, rows)
        isempty(protocol_rows) && continue
        plot_handle = plot(
            title = "Final reward (last $FINAL_WINDOW episodes) - " *
                    PROTOCOL_DISPLAY_NAMES[protocol],
            xlabel = "Configuration",
            ylabel = "Mean episode reward",
            legend = :bottomright,
            xticks = (
                1:length(CONFIG_NAMES),
                [CONFIG_DISPLAY_NAMES[name] for name in CONFIG_NAMES],
            ),
            framestyle = :box,
            gridalpha = 0.18,
            size = (850, 520),
            legend_background_color = :transparent,
            legend_foreground_color = :transparent,
        )
        for (index, config_name) in enumerate(CONFIG_NAMES)
            values = [
                row.final_reward_mean
                for row in protocol_rows
                if row.config === config_name
            ]
            isempty(values) && continue
            positions = index .+ collect(range(-0.07, 0.07; length = length(values)))
            scatter!(
                plot_handle,
                positions,
                values;
                color = CONFIG_COLORS[config_name],
                label = index == 1 ? "Individual runs" : "",
                markerstrokecolor = :white,
                markerstrokewidth = 0.6,
                markersize = 6,
            )
            scatter!(
                plot_handle,
                [index],
                [mean(values)];
                color = CONFIG_COLORS[config_name],
                label = index == 1 ? "Arithmetic mean" : "",
                marker = :diamond,
                markerstrokecolor = :black,
                markerstrokewidth = 1.0,
                markersize = 9,
            )
        end
        output = joinpath(output_directory, "final_performance_$(protocol).svg")
        savefig(plot_handle, output)
        push!(outputs, output)
    end
    return outputs
end

function plot_runtimes(rows, output_directory)
    outputs = String[]
    for protocol in PROTOCOLS
        protocol_rows = filter(row -> row.protocol === protocol, rows)
        isempty(protocol_rows) && continue
        plot_handle = plot(
            title = "Training runtime - $(PROTOCOL_DISPLAY_NAMES[protocol])",
            xlabel = "Configuration",
            ylabel = "Hours",
            legend = :bottomright,
            xticks = (
                1:length(CONFIG_NAMES),
                [CONFIG_DISPLAY_NAMES[name] for name in CONFIG_NAMES],
            ),
            framestyle = :box,
            gridalpha = 0.18,
            size = (850, 520),
            legend_background_color = :transparent,
            legend_foreground_color = :transparent,
        )
        for (index, config_name) in enumerate(CONFIG_NAMES)
            values = [
                row.elapsed_seconds / 3600
                for row in protocol_rows
                if row.config === config_name
            ]
            isempty(values) && continue
            positions = index .+ collect(range(-0.07, 0.07; length = length(values)))
            scatter!(
                plot_handle,
                positions,
                values;
                color = CONFIG_COLORS[config_name],
                label = index == 1 ? "Individual runs" : "",
                markerstrokecolor = :white,
                markerstrokewidth = 0.6,
                markersize = 6,
            )
            scatter!(
                plot_handle,
                [index],
                [mean(values)];
                color = CONFIG_COLORS[config_name],
                label = index == 1 ? "Arithmetic mean" : "",
                marker = :diamond,
                markerstrokecolor = :black,
                markerstrokewidth = 1.0,
                markersize = 9,
            )
        end
        output = joinpath(output_directory, "runtimes_$(protocol).svg")
        savefig(plot_handle, output)
        push!(outputs, output)
    end
    return outputs
end

function main(arguments = ARGS)
    options = parse_arguments(arguments)
    isnothing(options) && return
    results_directory = options["results_dir"]
    output_directory = options["output_dir"]
    mkpath(output_directory)

    records, problems = collect_records(results_directory)
    rows = metric_rows(records)
    csv_path = write_csv(joinpath(output_directory, "metrics.csv"), rows)

    for protocol in PROTOCOLS
        for prefix in ("learning_curves", "final_performance", "runtimes")
            for extension in ("png", "svg")
                rm(joinpath(output_directory, "$(prefix)_$(protocol).$(extension)"); force = true)
            end
        end
    end

    plot_paths = vcat(
        plot_learning_curves(records, output_directory),
        plot_final_performance(rows, output_directory),
        plot_runtimes(rows, output_directory),
    )
    summary_path = joinpath(output_directory, "summary.jld2")
    JLD2.jldsave(
        summary_path;
        schema_version = 1,
        collected_at = string(now()),
        results_directory = results_directory,
        complete_result_count = length(records),
        maximum_design_result_count = 30,
        problems = problems,
        metrics = rows,
        generated_plots = plot_paths,
    )

    println("Collected $(length(records)) complete result file(s).")
    println("Metrics: $csv_path")
    println("Summary: $summary_path")
    for path in plot_paths
        println("Plot: $path")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
