include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining
using Dates

function usage()
    println(
        """
        Usage: julia --project=. Revision/MAT_expert_training/run_test_worker.jl [options]

        Options:
          --results-dir PATH       Expert-training result root.
          --distillation-experts-dir PATH
                                   Existing tracked expert root to replace.
          --poll-seconds N         Wait interval; default: 60.
          --max-wait-hours N       Maximum wait; default: 720 (30 days).
          --help                   Show this message.
        """,
    )
end

function parse_options(arguments)
    options = Dict{String, Any}(
        "results_dir" => MATExpertTraining.DEFAULT_RESULTS_DIRECTORY,
        "distillation_experts_dir" => MATExpertTraining.DEFAULT_DISTILLATION_EXPERT_DIRECTORY,
        "poll_seconds" => 60.0,
        "max_wait_hours" => 720.0,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            exit(0)
        elseif argument in ("--results-dir", "--results_dir", "--distillation-experts-dir",
                            "--distillation_experts_dir", "--poll-seconds",
                            "--poll_seconds", "--max-wait-hours", "--max_wait_hours")
            index < length(arguments) || error("Missing value after $argument.")
            key = replace(argument[3:end], '-' => '_')
            options[key] = arguments[index + 1]
            index += 2
            continue
        else
            error("Unknown argument: $argument")
        end
    end
    options["poll_seconds"] = parse(Float64, string(options["poll_seconds"]))
    options["max_wait_hours"] = parse(Float64, string(options["max_wait_hours"]))
    options["poll_seconds"] > 0 || error("--poll-seconds must be positive.")
    options["max_wait_hours"] > 0 || error("--max-wait-hours must be positive.")
    return options
end

function main(arguments)
    options = parse_options(arguments)
    results_directory = string(options["results_dir"])
    poll_seconds = Float64(options["poll_seconds"])
    deadline = time() + 3600 * Float64(options["max_wait_hours"])
    pending = Set((:fixed, :varying))

    println("Test worker waiting for Fixed and Varying expert candidates.")
    flush(stdout)
    while !isempty(pending)
        for protocol in collect(pending)
            if MATExpertTraining.test_complete(results_directory, protocol)
                println("[$(now())] Existing complete $protocol test result detected.")
                delete!(pending, protocol)
            elseif MATExpertTraining.protocol_ready_for_test(results_directory, protocol)
                println(
                    "[$(now())] $protocol winner and all ten final checkpoints are ready; " *
                    "starting test-set evaluation.",
                )
                flush(stdout)
                MATExpertTraining.run_test_protocol_worker(; protocol, results_directory)
                delete!(pending, protocol)
            end
        end
        isempty(pending) && break
        time() <= deadline || error("Timed out waiting for protocols: $(join(sort!(string.(collect(pending))), ", ")).")
        println("[$(now())] Waiting for: $(join(sort!(string.(collect(pending))), ", ")).")
        flush(stdout)
        sleep(poll_seconds)
    end
    MATExpertTraining.publish_distillation_experts!(
        ;
        results_directory,
        distillation_expert_directory = string(options["distillation_experts_dir"]),
    )
    println("Fixed and Varying test-set evaluations are complete.")
end

main(ARGS)
