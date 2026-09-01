using JLD2

include(joinpath(@__DIR__, "NoiseStudy.jl"))
using .NoiseStudy

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function usage()
    println("""
    Usage: julia --startup-file=no --project=. Revision/Noise_Study/prepare_manifest.jl [options]

      --protocol fixed|varying
      --experiment-id ID
      --output PATH
      --package-results PATH
      --go-study-results PATH
      --baseline-results PATH
      --distillation-root PATH
      --print-jobs
      --help

    Building a manifest computes exact protocol-specific channel standard
    deviations from the complete distillation training corpus. --print-jobs
    only prints the fixed 30-worker grid and does not read result artifacts.
    """)
end

function parse_options(arguments)
    defaults = source_defaults(PROJECT_ROOT)
    values = Dict{String, Any}(
        "protocol" => nothing,
        "experiment_id" => nothing,
        "output" => nothing,
        "package_results" => nothing,
        "go_study_results" => defaults.go_study_results,
        "baseline_results" => defaults.baseline_results,
        "distillation_root" => defaults.distillation_root,
        "print_jobs" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--print-jobs"
            values["print_jobs"] = true
            index += 1
            continue
        end
        startswith(argument, "--") || error("Unknown argument '$argument'.")
        index < length(arguments) || error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    isnothing(values["protocol"]) && error("--protocol is required.")
    protocol = normalize_protocol(values["protocol"])
    package_default = protocol === :fixed ? defaults.package7_results : defaults.package8_results
    package_results = abspath(isnothing(values["package_results"]) ? package_default : string(values["package_results"]))
    if Bool(values["print_jobs"])
        return (;
            protocol,
            print_jobs = true,
            experiment_id = isnothing(values["experiment_id"]) ? "preview" : string(values["experiment_id"]),
            output = nothing,
            package_results,
            go_study_results = abspath(string(values["go_study_results"])),
            baseline_results = abspath(string(values["baseline_results"])),
            distillation_root = abspath(string(values["distillation_root"])),
        )
    end
    isnothing(values["experiment_id"]) && error("--experiment-id is required.")
    isnothing(values["output"]) && error("--output is required.")
    experiment_id = strip(string(values["experiment_id"]))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", experiment_id) || error("Invalid experiment ID '$experiment_id'.")
    return (;
        protocol,
        print_jobs = false,
        experiment_id,
        output = abspath(string(values["output"])),
        package_results,
        go_study_results = abspath(string(values["go_study_results"])),
        baseline_results = abspath(string(values["baseline_results"])),
        distillation_root = abspath(string(values["distillation_root"])),
    )
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    if options.print_jobs
        for job in job_records(options.protocol)
            println("$(job.controller)\t$(job.noise_level)\t$(job.level_tag)\t$(job.replicate_count)")
        end
        return nothing
    end
    manifest = build_protocol_manifest(
        options.output,
        options.protocol;
        package_results = options.package_results,
        go_study_results = options.go_study_results,
        baseline_results = options.baseline_results,
        distillation_root = options.distillation_root,
        experiment_id = options.experiment_id,
    )
    println("Wrote $(options.protocol) Noise-Study manifest: $(manifest.path)")
    println("Manifest fingerprint: $(manifest.fingerprint)")
    for controller in manifest.controllers
        println(
            "  $(controller.kind): $(controller.configuration), " *
            "active_inputs=$(controller.active_inputs), validation_mse=$(controller.validation_matching)",
        )
    end
    println("Channel scales (b, w, u): $(join(manifest.scales.scales, ", "))")
    return manifest.path
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
