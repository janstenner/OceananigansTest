include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining
using Dates
using Flux
using JLD2
using PlotlyJS
using Printf
using Statistics

const DEFAULT_OUTPUT_ROOT = joinpath(
    @__DIR__,
    "results",
    "package4_source_candidate_test",
)
const DEFAULT_EXPERT_RESULTS_DIRECTORY = joinpath(@__DIR__, "results")

function usage()
    println(
        """
        Usage: julia --startup-file=no --project=. \\
          Revision/MAT_expert_training/evaluate_package4_candidate.jl [options]

        Evaluate either the validation-rank-1 Package-4 MAT source checkpoint
        or the MAT-expert-training threshold winner with deterministic mean
        actions and record state_Nu(env) at every test step.
        Run Fixed and Varying in separate Julia processes.

        Options:
          --protocol fixed|varying  Protocol to evaluate (required).
          --selection package4-rank1|threshold
                                   Candidate selection; default: package4-rank1.
          --source-results-dir PATH MAT_IPPO_Comparison result root.
          --expert-results-dir PATH MAT_expert_training result root containing
                                   <protocol>/candidate.jld2 and best_so_far.jld2.
          --output-root PATH        Output root; protocol is appended.
          --help                    Show this message.
        """,
    )
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "protocol" => nothing,
        "selection" => "package4-rank1",
        "source_results_dir" => MATExpertTraining.DEFAULT_SOURCE_RESULTS_DIRECTORY,
        "expert_results_dir" => DEFAULT_EXPERT_RESULTS_DIRECTORY,
        "output_root" => nothing,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            exit(0)
        elseif argument in (
            "--protocol",
            "--selection",
            "--source-results-dir",
            "--source_results_dir",
            "--expert-results-dir",
            "--expert_results_dir",
            "--output-root",
            "--output_root",
        )
            index < length(arguments) || error("Missing value after $argument.")
            options[replace(argument[3:end], '-' => '_')] = arguments[index + 1]
            index += 2
            continue
        end
        error("Unknown argument: $argument")
    end
    isnothing(options["protocol"]) && error("--protocol fixed|varying is required.")
    protocol = Symbol(lowercase(string(options["protocol"])))
    protocol in (:fixed, :varying) || error("Unknown protocol: $protocol")
    selection = Symbol(replace(lowercase(string(options["selection"])), '-' => '_'))
    selection in (:package4_rank1, :threshold) || error(
        "Unknown --selection $(options["selection"]); use package4-rank1 or threshold.",
    )
    output_root = if isnothing(options["output_root"])
        selection === :package4_rank1 ? DEFAULT_OUTPUT_ROOT :
            joinpath(@__DIR__, "results", "threshold_candidate_test")
    else
        string(options["output_root"])
    end
    return (;
        protocol,
        selection,
        source_results_directory = abspath(string(options["source_results_dir"])),
        expert_results_directory = abspath(string(options["expert_results_dir"])),
        output_root = abspath(output_root),
    )
end

rank_one_candidate(protocol) = only(
    candidate for candidate in MATExpertTraining.CANDIDATES
    if candidate.protocol === protocol && candidate.rank == 1
)

function threshold_candidate(protocol, results_directory)
    manifest_path = joinpath(results_directory, string(protocol), "candidate.jld2")
    checkpoint_path = joinpath(results_directory, string(protocol), "best_so_far.jld2")
    isfile(manifest_path) || error("Missing threshold candidate manifest: $manifest_path")
    isfile(checkpoint_path) || error("Missing threshold checkpoint: $checkpoint_path")
    manifest = JLD2.load(manifest_path)
    string(manifest["selection_mode"]) == "threshold" || error(
        "Candidate manifest is not a threshold selection: $manifest_path",
    )
    Bool(manifest["threshold_reached"]) || error(
        "Candidate manifest did not reach the threshold: $manifest_path",
    )
    Symbol(manifest["protocol"]) === protocol || error("Manifest protocol mismatch.")
    run_id = string(manifest["winner_run_id"])
    candidate = only(
        item for item in MATExpertTraining.CANDIDATES
        if item.protocol === protocol && item.run_id == run_id
    )
    checkpoint = JLD2.jldopen(checkpoint_path, "r") do file
        (
            status = string(read(file, "status")),
            protocol = Symbol(read(file, "protocol")),
            run_id = string(read(file, "run_id")),
            additional_episodes = Int(read(file, "additional_episodes")),
            total_episodes = Int(read(file, "total_episodes")),
            criterion_value = Float64(read(file, "criterion_value")),
            threshold_reached = Bool(read(file, "threshold_reached")),
        )
    end
    checkpoint.status == "best_so_far" || error("Unexpected checkpoint status.")
    checkpoint.protocol === protocol || error("Threshold checkpoint protocol mismatch.")
    checkpoint.run_id == run_id || error("Threshold checkpoint run-ID mismatch.")
    checkpoint.threshold_reached || error("Threshold checkpoint did not reach threshold.")
    checkpoint.additional_episodes == Int(manifest["additional_episodes"]) || error(
        "Threshold checkpoint/manifest episode mismatch.",
    )
    isapprox(
        checkpoint.criterion_value,
        Float64(manifest["criterion_value"]);
        atol = 1e-10,
        rtol = 0.0,
    ) || error("Threshold checkpoint/manifest criterion mismatch.")
    return (; candidate, manifest_path, checkpoint_path, checkpoint...)
end

function load_cached_record(cache_path, checkpoint_sha256, case)
    isfile(cache_path) || return nothing
    cached = JLD2.load(cache_path)
    string(cached["status"]) == "complete" || return nothing
    string(cached["checkpoint_sha256"]) == checkpoint_sha256 || return nothing
    haskey(cached, "global_nusselt") || return nothing
    return (;
        case_id = case.case_id,
        choice = case.choice,
        rewards = Float64.(cached["rewards"]),
        global_nusselt = Float64.(cached["global_nusselt"]),
    )
end

function log_state_nusselt(protocol, record; source)
    for (step, value) in enumerate(record.global_nusselt)
        println(
            "state_Nu protocol=$protocol case=$(record.case_id) " *
            "step=$step value=$(@sprintf("%.12f", value)) source=$source",
        )
    end
    flush(stdout)
end

function save_episode(cache_path, protocol, candidate, record, checkpoint_path,
                      checkpoint_sha256)
    rewards = record.rewards
    global_nusselt = record.global_nusselt
    MATExpertTraining.atomic_jldsave(
        cache_path;
        status = "complete",
        protocol,
        run_id = candidate.run_id,
        validation_rank = candidate.rank,
        validation_score = candidate.validation_score,
        checkpoint_path,
        checkpoint_sha256,
        case_id = record.case_id,
        choice = record.choice,
        policy = "deterministic_mean_action",
        rewards,
        reward_score = sum(rewards),
        global_nusselt,
        sum_state_nusselt = sum(global_nusselt),
        mean_state_nusselt = mean(global_nusselt),
        completed_at = string(now()),
    )
end

function curve_plot(record, protocol, selection_title, output_directory)
    plot_handle = Plot(
        scatter(
            x = collect(eachindex(record.global_nusselt)),
            y = record.global_nusselt,
            mode = "lines",
            line = attr(color = "#B2182B", width = 3),
            name = "state_Nu(env)",
        ),
        Layout(
            template = "plotly_white",
            title = attr(
                text = "$selection_title — $(record.case_id)",
                x = 0.5,
                xanchor = "center",
            ),
            paper_bgcolor = "white",
            plot_bgcolor = "white",
            width = 1000,
            height = 600,
            margin = attr(l = 95, r = 35, t = 80, b = 80),
            xaxis = attr(title = "Control step", gridcolor = "#E6E6E6"),
            yaxis = attr(
                title = "Full-state global Nusselt number state_Nu(env)",
                gridcolor = "#E6E6E6",
            ),
            showlegend = false,
        ),
    )
    stem = joinpath(output_directory, record.case_id * "_state_nu")
    PlotlyJS.savefig(plot_handle, stem * ".svg"; width = 1000, height = 600)
    PlotlyJS.savefig(plot_handle, stem * ".png"; width = 1000, height = 600)
    return (; svg = stem * ".svg", png = stem * ".png")
end

function box_plot(records, selection_title, output_directory)
    sums = sum.(getproperty.(records, :global_nusselt))
    labels = getproperty.(records, :case_id)
    plot_handle = Plot(
        box(
            y = sums,
            text = labels,
            name = "8 deterministic test episodes",
            boxpoints = "all",
            jitter = 0.35,
            pointpos = 0.0,
            marker = attr(color = "#2166AC", size = 9),
            line = attr(color = "#2166AC"),
            hovertemplate = "%{text}<br>sum(state_Nu)=%{y:.6f}<extra></extra>",
        ),
        Layout(
            template = "plotly_white",
            title = attr(
                text = "$selection_title — deterministic test distribution",
                x = 0.5,
                xanchor = "center",
            ),
            paper_bgcolor = "white",
            plot_bgcolor = "white",
            width = 800,
            height = 650,
            margin = attr(l = 110, r = 45, t = 80, b = 70),
            xaxis = attr(showticklabels = false, gridcolor = "#E6E6E6"),
            yaxis = attr(
                title = "sum(state_Nu(env)) over 200 control steps (lower is better)",
                gridcolor = "#E6E6E6",
            ),
            showlegend = false,
        ),
    )
    stem = joinpath(output_directory, "sum_state_nu_boxplot")
    PlotlyJS.savefig(plot_handle, stem * ".svg"; width = 800, height = 650)
    PlotlyJS.savefig(plot_handle, stem * ".png"; width = 800, height = 650)
    return (; svg = stem * ".svg", png = stem * ".png")
end

function write_step_csv(path, records)
    open(path, "w") do io
        println(io, "case_id,base_seed,mirror,offset,step,state_nu,reward")
        for record in records
            choice = record.choice
            for step in eachindex(record.global_nusselt)
                println(io, join((
                    record.case_id,
                    isnothing(choice) ? "" : choice.base_seed,
                    isnothing(choice) ? "" : choice.mirror,
                    isnothing(choice) ? "" : choice.offset,
                    step,
                    record.global_nusselt[step],
                    record.rewards[step],
                ), ','))
            end
        end
    end
    return path
end

function write_summary_csv(path, records)
    open(path, "w") do io
        println(
            io,
            "case_id,base_seed,mirror,offset,sum_state_nu,mean_state_nu,reward_score",
        )
        for record in records
            choice = record.choice
            println(io, join((
                record.case_id,
                isnothing(choice) ? "" : choice.base_seed,
                isnothing(choice) ? "" : choice.mirror,
                isnothing(choice) ? "" : choice.offset,
                sum(record.global_nusselt),
                mean(record.global_nusselt),
                sum(record.rewards),
            ), ','))
        end
    end
    return path
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    protocol = options.protocol
    threshold = options.selection === :threshold ? threshold_candidate(
        protocol,
        options.expert_results_directory,
    ) : nothing
    candidate = isnothing(threshold) ? rank_one_candidate(protocol) : threshold.candidate
    verified_source = MATExpertTraining.verified_candidate_record(
        candidate, options.source_results_directory,
    )
    checkpoint_path = isnothing(threshold) ? verified_source.checkpoint_path :
        abspath(threshold.checkpoint_path)
    checkpoint_sha256 = MATExpertTraining.source_hash(checkpoint_path)
    selection_title = options.selection === :package4_rank1 ?
        "$(uppercasefirst(string(protocol)))-IC MAT validation rank 1" :
        "$(uppercasefirst(string(protocol)))-IC MAT threshold winner"
    output_directory = joinpath(options.output_root, string(protocol))
    episode_directory = joinpath(output_directory, "episodes")
    curve_directory = joinpath(output_directory, "state_nu_curves")
    runtime_directory = joinpath(output_directory, "runtime")
    mkpath.((episode_directory, curve_directory, runtime_directory))

    println(
        "Evaluating $protocol $(options.selection) MAT candidate $(candidate.run_id) " *
        "(validation=$(candidate.validation_score), checkpoint_sha256=$checkpoint_sha256).",
    )
    flush(stdout)
    MATExpertTraining.include_run_file!(candidate, runtime_directory)
    loaded_agent = JLD2.load(checkpoint_path, "agent")
    Flux.testmode!(loaded_agent.policy)
    Core.eval(MATExpertTraining, :(agent = $loaded_agent))

    records = NamedTuple[]
    for case in MATExpertTraining.test_cases(protocol)
        cache_path = joinpath(episode_directory, case.case_id * ".jld2")
        cached = load_cached_record(cache_path, checkpoint_sha256, case)
        if isnothing(cached)
            rollout = Base.invokelatest(
                MATExpertTraining.deterministic_test_rollout,
                protocol,
                case.choice,
            )
            record = (;
                case_id = case.case_id,
                choice = case.choice,
                rewards = rollout.rewards,
                global_nusselt = rollout.global_nusselt,
            )
            save_episode(
                cache_path,
                protocol,
                candidate,
                record,
                checkpoint_path,
                checkpoint_sha256,
            )
            log_state_nusselt(protocol, record; source = "rollout")
        else
            record = cached
            log_state_nusselt(protocol, record; source = "cache")
        end
        push!(records, record)
        println(
            "Completed $(case.case_id): sum(state_Nu)=$(sum(record.global_nusselt)), " *
            "mean(state_Nu)=$(mean(record.global_nusselt)), " *
            "reward_score=$(sum(record.rewards)).",
        )
        flush(stdout)
    end

    curves = [curve_plot(record, protocol, selection_title, curve_directory) for record in records]
    boxplot = protocol === :varying ? box_plot(records, selection_title, output_directory) : nothing
    step_csv = write_step_csv(joinpath(output_directory, "state_nu_steps.csv"), records)
    summary_csv = write_summary_csv(joinpath(output_directory, "summary.csv"), records)
    sums = sum.(getproperty.(records, :global_nusselt))
    means = mean.(getproperty.(records, :global_nusselt))
    reward_scores = sum.(getproperty.(records, :rewards))
    summary_path = joinpath(output_directory, "summary.jld2")
    MATExpertTraining.atomic_jldsave(
        summary_path;
        status = "complete",
        protocol,
        selection = options.selection,
        run_id = candidate.run_id,
        validation_rank = candidate.rank,
        validation_score = candidate.validation_score,
        checkpoint_path,
        checkpoint_sha256,
        threshold_manifest_path = isnothing(threshold) ? "" : abspath(threshold.manifest_path),
        additional_episodes = isnothing(threshold) ? 0 : threshold.additional_episodes,
        total_episodes = isnothing(threshold) ?
            MATExpertTraining.ORIGINAL_EPISODES[protocol] : threshold.total_episodes,
        threshold_criterion_value = isnothing(threshold) ? NaN : threshold.criterion_value,
        policy = "deterministic_mean_action",
        cases = records,
        sum_state_nusselt = sums,
        mean_sum_state_nusselt = mean(sums),
        median_sum_state_nusselt = median(sums),
        mean_state_nusselt_by_case = means,
        reward_scores,
        mean_reward_score = mean(reward_scores),
        state_nu_step_csv = abspath(step_csv),
        summary_csv = abspath(summary_csv),
        curve_svgs = abspath.(getproperty.(curves, :svg)),
        curve_pngs = abspath.(getproperty.(curves, :png)),
        boxplot_svg = isnothing(boxplot) ? "" : abspath(boxplot.svg),
        boxplot_png = isnothing(boxplot) ? "" : abspath(boxplot.png),
        completed_at = string(now()),
    )
    println(
        "Complete: protocol=$protocol cases=$(length(records)) " *
        "mean_sum_state_Nu=$(mean(sums)) median_sum_state_Nu=$(median(sums)) " *
        "mean_reward_score=$(mean(reward_scores)) summary=$summary_path",
    )
    return summary_path
end

main()
