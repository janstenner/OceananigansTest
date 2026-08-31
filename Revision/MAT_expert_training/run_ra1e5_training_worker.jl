include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining

function usage()
    println(
        """
        Usage: julia --project=. Revision/MAT_expert_training/run_ra1e5_training_worker.jl [options]

        Required stop rule (choose exactly one):
          --episodes N              Train every run for exactly N episodes.
          --threshold VALUE         Stop all runs after the first rolling-100 mean > VALUE.

        Options:
          --run-index N             One-based run index (required unless --prepare-only).
          --runs N                  Number of independent runs (default: 10).
          --master-seed N           Seed used to derive all run/IC seeds.
          --results-dir PATH        Result root (default: results_ra1e5).
          --prepare-only            Freeze/verify the experiment manifest; do not train.
          --preview                 Print and validate the plan without writing or training.
          --help                    Show this message.
        """,
    )
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "run_index" => nothing,
        "runs" => 10,
        "master_seed" => MATExpertTraining.DEFAULT_RA1E5_MASTER_SEED,
        "results_dir" => MATExpertTraining.DEFAULT_RA1E5_RESULTS_DIRECTORY,
        "episodes" => nothing,
        "threshold" => nothing,
        "prepare_only" => false,
        "preview" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            exit(0)
        elseif argument in ("--prepare-only", "--prepare_only", "--preview")
            options[replace(argument[3:end], '-' => '_')] = true
            index += 1
        elseif argument in ("--run-index", "--run_index", "--runs", "--master-seed",
                            "--master_seed", "--results-dir", "--results_dir",
                            "--episodes", "--threshold")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], '-' => '_')
            raw = arguments[index + 1]
            if key in ("run_index", "runs", "master_seed", "episodes")
                options[key] = parse(Int, raw)
            elseif key == "threshold"
                options[key] = parse(Float64, raw)
            else
                options[key] = raw
            end
            index += 2
        else
            error("Unknown argument: $argument")
        end
    end
    options["runs"] > 0 || error("--runs must be positive.")
    selected_stop_rules = count(!isnothing, (options["episodes"], options["threshold"]))
    selected_stop_rules == 1 || error(
        "Choose exactly one stop rule: --episodes N or --threshold VALUE.",
    )
    if !options["prepare_only"] && !options["preview"]
        isnothing(options["run_index"]) && error("--run-index is required for training.")
        1 <= options["run_index"] <= options["runs"] || error(
            "--run-index must lie in 1:$(options["runs"]).",
        )
    end
    return options
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    stop_mode = isnothing(options["episodes"]) ? :threshold : :episodes
    episode_limit = isnothing(options["episodes"]) ? 0 : options["episodes"]
    threshold = isnothing(options["threshold"]) ? 0.0 : options["threshold"]
    if options["prepare_only"] || options["preview"]
        experiment = MATExpertTraining.freeze_ra1e5_experiment!(
            results_directory = options["results_dir"],
            master_seed = options["master_seed"],
            run_count = options["runs"],
            stop_mode = stop_mode,
            episode_limit = episode_limit,
            threshold = threshold,
            preview = options["preview"],
        )
        println(options["preview"] ? "Ra=1e5 experiment preview:" : "Frozen Ra=1e5 experiment:")
        println(
            "  runs=$(experiment.run_count), master_seed=$(experiment.master_seed), " *
            "stop_mode=$(experiment.stop_mode), " *
            (experiment.stop_mode === :episodes ?
                "episodes=$(experiment.episode_limit)" : "threshold=$(experiment.threshold)"),
        )
        println("  corpus=$(experiment.corpus_path)")
        println("  corpus_sha256=$(experiment.corpus_sha256)")
        println("  config_fingerprint=$(experiment.config_fingerprint)")
        for candidate in experiment.candidates
            println(
                "  r$(lpad(string(candidate.run_index), 2, '0')): " *
                "$(candidate.run_id), run_seed=$(candidate.run_seed), ic_seed=$(candidate.ic_seed)",
            )
        end
        return experiment
    end
    return MATExpertTraining.run_ra1e5_training_worker(
        run_index = options["run_index"],
        results_directory = options["results_dir"],
        master_seed = options["master_seed"],
        run_count = options["runs"],
        stop_mode = stop_mode,
        episode_limit = episode_limit,
        threshold = threshold,
    )
end

main()
