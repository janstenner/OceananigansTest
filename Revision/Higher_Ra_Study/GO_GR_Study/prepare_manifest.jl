using Dates

include(joinpath(@__DIR__, "HigherRaGOStudy.jl"))
using .HigherRaGOStudy

function parse_arguments(arguments)
    output = nothing
    experiment_id = nothing
    study_value = nothing
    selection = "all"
    strengths = Float64[]
    thresholds = Float64[]
    results_root = DEFAULT_RESULTS_ROOT
    print_variants = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            println("Usage: prepare_manifest.jl --study ra5e4|ra1e5 [--print-variants] [--output PATH --experiment-id ID] [--config all|NAME] [--strength VALUE ...] [--threshold VALUE ...] [--results-dir PATH]")
            return nothing
        elseif argument == "--print-variants"
            print_variants = true
            index += 1
            continue
        end
        index < length(arguments) || error("Missing value after $argument.")
        value = arguments[index + 1]
        if argument == "--output"
            output = value
        elseif argument == "--experiment-id"
            experiment_id = value
        elseif argument == "--study"
            study_value = value
        elseif argument == "--config"
            selection = value
        elseif argument == "--strength"
            push!(strengths, parse(Float64, value))
        elseif argument == "--threshold"
            push!(thresholds, parse(Float64, value))
        elseif argument == "--results-dir"
            results_root = value
        else
            error("Unknown option '$argument'.")
        end
        index += 2
    end
    isnothing(study_value) && error("--study is required.")
    study_tag = normalize_study(study_value)
    lowercase(string(selection)) == "all" && !isempty(thresholds) && error(
        "Explicit mask thresholds require exactly one --config.",
    )
    if print_variants
        return (; output = nothing, experiment_id = nothing, study = study_tag,
                selection, strengths, thresholds, results_root = abspath(results_root),
                print_variants)
    end
    isnothing(output) && error("--output is required.")
    isnothing(experiment_id) && error("--experiment-id is required.")
    return (; output = abspath(output),
            experiment_id = normalize_experiment_id(experiment_id), study = study_tag,
            selection, strengths, thresholds, results_root = abspath(results_root),
            print_variants)
end

function prepare_manifest(options)
    variants = selected_variants(options.selection, options.strengths)
    jobs = study_jobs(options.study, options.experiment_id, options.selection, options.strengths)
    thresholds = resolved_mask_thresholds(options.thresholds)
    sources = source_manifest(options.study)
    length(unique(job.id for job in jobs)) == length(jobs) || error("Duplicate Higher-Ra run IDs.")
    length(unique(job.relative_path for job in jobs)) == length(jobs) || error("Duplicate Higher-Ra run paths.")
    atomic_save(
        options.output;
        schema_version = HR_SCHEMA_VERSION,
        experiment = :higher_ra_go_gr_comparison,
        experiment_id = options.experiment_id,
        study = options.study,
        protocol = sources.protocol,
        rayleigh = sources.rayleigh,
        training_data_split = :train,
        validation_data_split = :validation,
        terminal_data_split = :test,
        master_seed = HR_MASTER_SEED,
        seed_plan = [seed_plan(replicate) for replicate in HR_REPLICATES],
        mask_threshold_values = thresholds,
        quality_threshold_values = collect(HR_QUALITY_THRESHOLDS),
        threshold_mode = :absolute,
        threshold_importance_mode = :max_input_l1,
        threshold_group_aggregation = :maximum,
        threshold_minimum_active_groups = 1,
        variants,
        jobs,
        sources,
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
            println("Wrote $(length(jobs)) jobs for $(length(variants)) $(options.study) variant(s).")
        end
    end
end
