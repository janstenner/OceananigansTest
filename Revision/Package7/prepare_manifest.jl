using Dates

include(joinpath(@__DIR__, "Package7Study.jl"))
using .Package7Study

function parse_arguments(arguments)
    output = nothing
    experiment_id = nothing
    selection = "all"
    strengths = Float64[]
    results_root = joinpath(@__DIR__, "results")
    print_variants = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        argument == "--help" && begin
            println("Usage: prepare_manifest.jl [--print-variants] [--output PATH --experiment-id ID] [--config all|NAME] [--strength VALUE ...] [--results-dir PATH]")
            return nothing
        end
        if argument == "--print-variants"
            print_variants = true
            index += 1
            continue
        end
        index == length(arguments) && error("Missing value after $argument.")
        value = arguments[index + 1]
        if argument == "--output"
            output = value
        elseif argument == "--experiment-id"
            experiment_id = value
        elseif argument == "--config"
            selection = value
        elseif argument == "--strength"
            push!(strengths, parse(Float64, value))
        elseif argument == "--results-dir"
            results_root = value
        else
            error("Unknown option '$argument'.")
        end
        index += 2
    end
    if print_variants
        return (; output = nothing, experiment_id = nothing, selection, strengths,
                results_root = abspath(results_root), print_variants)
    end
    isnothing(output) && error("--output is required.")
    isnothing(experiment_id) && error("--experiment-id is required.")
    return (; output = abspath(output), experiment_id = normalize_experiment_id(experiment_id),
            selection, strengths, results_root = abspath(results_root), print_variants)
end

function prepare_manifest(options)
    variants = selected_variants(options.selection, options.strengths)
    jobs = study_jobs(options.experiment_id, options.selection, options.strengths)
    length(unique(job.id for job in jobs)) == length(jobs) || error("Duplicate Package-7 run IDs.")
    length(unique(job.relative_path for job in jobs)) == length(jobs) || error("Duplicate Package-7 run paths.")
    atomic_save(
        options.output;
        schema_version = P7_SCHEMA_VERSION,
        experiment = :package7_fixed_regularizer_comparison,
        experiment_id = options.experiment_id,
        protocol = :fixed,
        master_seed = P7_MASTER_SEED,
        seed_plan = [seed_plan(replicate) for replicate in P7_REPLICATES],
        threshold_values = collect(P7_THRESHOLDS),
        threshold_mode = :absolute,
        threshold_importance_mode = :max_input_l1,
        threshold_group_aggregation = :maximum,
        threshold_minimum_active_groups = 1,
        variants,
        jobs,
        results_root = options.results_root,
        created_at = string(Dates.now()),
    )
    return (variants, jobs)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    options = parse_arguments(ARGS)
    if !isnothing(options)
        if options.print_variants
            for variant in selected_variants(options.selection, options.strengths)
                println("$(variant.name)\t$(variant.strength)")
            end
        else
            variants, jobs = prepare_manifest(options)
            println("Wrote $(length(jobs)) training jobs for $(length(variants)) Package-7 variant(s).")
        end
    end
end
