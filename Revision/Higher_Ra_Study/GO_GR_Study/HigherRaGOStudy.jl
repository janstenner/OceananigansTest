module HigherRaGOStudy

using Dates
using JLD2
using Printf
using SHA
using StableRNGs

export HR_SCHEMA_VERSION, HR_MASTER_SEED, HR_UPDATES, HR_BATCH_SIZE,
       HR_VALIDATION_BATCH_SIZE, HR_LEARNING_RATE, HR_EVALUATION_INTERVAL,
       HR_RESUME_INTERVAL, HR_GARBAGE_COLLECTION_INTERVAL, HR_REPLICATES,
       HR_MASK_THRESHOLDS, HR_QUALITY_THRESHOLDS, HR_CONFIGURATION_NAMES,
       HR_STRENGTH_GRIDS, HR_STUDIES, DEFAULT_RESULTS_ROOT, study,
       configuration, normalize_study, normalize_configuration,
       normalize_experiment_id, seed_plan, seed_plan_hash, selected_variants,
       resolved_mask_thresholds, study_jobs, job_for, run_directory,
       analysis_directory, status_path, analysis_status_path, atomic_save,
       load_status, write_status!, canonical_string, fingerprint, file_sha256,
       strength_tag, expected_evaluation_updates, expected_corpus_files,
       source_manifest

const HR_SCHEMA_VERSION = 1
const HR_MASTER_SEED = 20_260_904
const HR_UPDATES = 100_000
const HR_BATCH_SIZE = 100
const HR_VALIDATION_BATCH_SIZE = 512
const HR_LEARNING_RATE = 2e-4
const HR_EVALUATION_INTERVAL = 25
const HR_RESUME_INTERVAL = 100
const HR_GARBAGE_COLLECTION_INTERVAL = 5
const HR_REPLICATES = 1:3
const HR_MASK_THRESHOLDS = (0.0, 0.003, 0.006, 0.012)
const HR_QUALITY_THRESHOLDS = (0.03, 0.015, 0.0075)

const STUDY_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = normpath(joinpath(STUDY_DIRECTORY, "..", ".."))
const DEFAULT_RESULTS_ROOT = joinpath(@__DIR__, "results")

const HR_STUDIES = (
    ra5e4 = (
        tag = :ra5e4,
        label = "Ra=5e4",
        protocol = :varying_ra5e4,
        rayleigh = 5.0e4,
        run_file = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "VaryingIC_MAT_Ra5e4.jl"),
        state_corpus = joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus_Ra5e4.jld2"),
        distillation_root = joinpath(STUDY_DIRECTORY, "Distillation_Corpuses", "ra5e4", "worker_results"),
        expert = joinpath(STUDY_DIRECTORY, "experts", "ra5e4", "expert.jld2"),
        expert_baseline = joinpath(STUDY_DIRECTORY, "Baselines", "ra5e4", "expert.jld2"),
        unactuated_baseline = joinpath(STUDY_DIRECTORY, "Baselines", "ra5e4", "unactuated.jld2"),
    ),
    ra1e5 = (
        tag = :ra1e5,
        label = "Ra=1e5",
        protocol = :varying_ra1e5,
        rayleigh = 1.0e5,
        run_file = joinpath(PROJECT_ROOT, "Revision", "Run_Files", "VaryingIC_MAT_Ra1e5.jl"),
        state_corpus = joinpath(PROJECT_ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus_Ra1e5.jld2"),
        distillation_root = joinpath(STUDY_DIRECTORY, "Distillation_Corpuses", "ra1e5", "worker_results"),
        expert = joinpath(STUDY_DIRECTORY, "experts", "ra1e5", "expert.jld2"),
        expert_baseline = joinpath(STUDY_DIRECTORY, "Baselines", "ra1e5", "expert.jld2"),
        unactuated_baseline = joinpath(STUDY_DIRECTORY, "Baselines", "ra1e5", "unactuated.jld2"),
    ),
)

const HR_CONFIGURATION_NAMES = ("go-gc", "go-sc", "gr-gc", "gr-sc")

const HR_STRENGTH_GRIDS = Dict(
    "go-gc" => (0.00128, 0.0032, 0.008, 0.02, 0.05),
    "go-sc" => (0.00256, 0.0064, 0.016, 0.04, 0.1),
    "gr-gc" => (0.00000384, 0.0000096, 0.000024, 0.00006, 0.00015),
    "gr-sc" => (0.00000768, 0.0000192, 0.000048, 0.00012, 0.0003),
)

const HR_CONFIGURATIONS = Dict(
    "go-gc" => (method = :go, grouping = :grouped_channels, group_channels = true),
    "go-sc" => (method = :go, grouping = :separate_channels, group_channels = false),
    "gr-gc" => (method = :gr, grouping = :grouped_channels, group_channels = true),
    "gr-sc" => (method = :gr, grouping = :separate_channels, group_channels = false),
)

function normalize_study(value)::Symbol
    tag = Symbol(lowercase(replace(strip(string(value)), "-" => "")))
    tag = get(Dict(:ra50000 => :ra5e4, :ra100000 => :ra1e5), tag, tag)
    tag in keys(HR_STUDIES) || throw(ArgumentError(
        "Study must be ra5e4 or ra1e5, got '$value'.",
    ))
    return tag
end

study(value) = getproperty(HR_STUDIES, normalize_study(value))

function normalize_configuration(value)::String
    name = lowercase(strip(string(value)))
    haskey(HR_CONFIGURATIONS, name) || throw(ArgumentError(
        "Unknown Higher-Ra configuration '$value'. Available: $(join(HR_CONFIGURATION_NAMES, ", ")).",
    ))
    return name
end

configuration(value) = HR_CONFIGURATIONS[normalize_configuration(value)]

function normalize_experiment_id(value)::String
    identifier = strip(string(value))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", identifier) || throw(ArgumentError(
        "Experiment ID '$value' must contain only letters, digits, underscores, and hyphens.",
    ))
    return identifier
end

function resolved_mask_thresholds(values = Float64[])
    isempty(values) && return collect(HR_MASK_THRESHOLDS)
    custom = sort!(unique(Float64.(values)))
    all(value -> isfinite(value) && value > 0, custom) || throw(ArgumentError(
        "Custom mask thresholds must be finite and positive; native threshold 0.0 is automatic.",
    ))
    return vcat(0.0, custom)
end

function seed_plan(replicate::Integer)
    replicate in HR_REPLICATES || throw(ArgumentError("Replicate must be in 1:3."))
    planner = StableRNG(HR_MASTER_SEED)
    apprentice_seed = 0
    batch_seed = 0
    for _ in 1:replicate
        apprentice_seed = rand(planner, 1:2_000_000_000)
        batch_seed = rand(planner, 1:2_000_000_000)
    end
    return (; replicate = Int(replicate), apprentice_seed, batch_seed)
end

function canonical_string(value)
    if value isa AbstractDict
        entries = sort!(collect(pairs(value)); by = pair -> string(first(pair)))
        return "{" * join((canonical_string(first(entry)) * ":" * canonical_string(last(entry)) for entry in entries), ",") * "}"
    elseif value isa NamedTuple
        return canonical_string(Dict(pairs(value)))
    elseif value isa Tuple || value isa AbstractVector || value isa AbstractRange
        return "[" * join(canonical_string.(collect(value)), ",") * "]"
    elseif value isa Symbol
        return ":" * string(value)
    elseif value isa AbstractString
        return repr(String(value))
    elseif value === nothing
        return "nothing"
    end
    return repr(value)
end

fingerprint(value) = bytes2hex(SHA.sha256(codeunits(canonical_string(value))))
seed_plan_hash(replicate::Integer) = fingerprint(seed_plan(replicate))
file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function strength_tag(strength::Real)
    value = Float64(strength)
    isfinite(value) && value > 0 || throw(ArgumentError("Strength must be finite and positive."))
    canonical = lowercase(@sprintf("%.12g", value))
    return "s_" * replace(canonical, "." => "p", "+" => "", "-" => "m")
end

function selected_variants(selection = "all", strengths = Float64[])
    if lowercase(string(selection)) == "all"
        isempty(strengths) || throw(ArgumentError("Explicit strengths require exactly one --config."))
        return [(name = name, strength = strength) for name in HR_CONFIGURATION_NAMES for strength in HR_STRENGTH_GRIDS[name]]
    end
    name = normalize_configuration(selection)
    values = isempty(strengths) ? collect(HR_STRENGTH_GRIDS[name]) : unique(Float64.(strengths))
    all(value -> isfinite(value) && value > 0, values) || throw(ArgumentError(
        "Strengths must be finite and positive.",
    ))
    return [(name, strength = value) for value in values]
end

function job_for(study_value, experiment_id, configuration_name, strength::Real,
                 replicate::Integer; updates::Integer = HR_UPDATES)
    study_config = study(study_value)
    experiment = normalize_experiment_id(experiment_id)
    name = normalize_configuration(configuration_name)
    config = configuration(name)
    seeds = seed_plan(replicate)
    tag = strength_tag(strength)
    replicate_tag = @sprintf("r%02d", replicate)
    relative_path = joinpath(string(study_config.tag), experiment, name, tag, replicate_tag)
    return (
        study = study_config.tag,
        rayleigh = study_config.rayleigh,
        experiment_id = experiment,
        configuration = name,
        config...,
        regularization_strength = Float64(strength),
        strength_tag = tag,
        replicate = Int(replicate),
        seeds...,
        pairing_hash = seed_plan_hash(replicate),
        updates = Int(updates),
        id = "hr_$(study_config.tag)_$(experiment)_$(replace(name, "-" => "_"))_$(tag)_$(replicate_tag)",
        relative_path,
    )
end

function study_jobs(study_value, experiment_id, selection = "all", strengths = Float64[];
                    updates::Integer = HR_UPDATES)
    return [
        job_for(study_value, experiment_id, variant.name, variant.strength, replicate; updates)
        for variant in selected_variants(selection, strengths)
        for replicate in HR_REPLICATES
    ]
end

run_directory(results_root::AbstractString, job) = joinpath(abspath(results_root), job.relative_path)
analysis_directory(results_root::AbstractString, study_value, experiment_id, configuration_name) =
    joinpath(abspath(results_root), string(normalize_study(study_value)),
             normalize_experiment_id(experiment_id), normalize_configuration(configuration_name), "analysis")
status_path(results_root::AbstractString, job) = joinpath(run_directory(results_root, job), "status.jld2")
analysis_status_path(results_root::AbstractString, study_value, experiment_id, configuration_name) =
    joinpath(analysis_directory(results_root, study_value, experiment_id, configuration_name), "status.jld2")

function atomic_save(path::AbstractString; entries...)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid()).$(time_ns())"
    try
        JLD2.jldopen(temporary, "w") do file
            for (key, value) in pairs(entries)
                file[string(key)] = value
            end
        end
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return abspath(path)
end

function load_status(path::AbstractString)
    isfile(path) || return nothing
    return Dict{Symbol, Any}(Symbol(key) => value for (key, value) in JLD2.load(path))
end

write_status!(path::AbstractString; entries...) =
    atomic_save(path; schema_version = HR_SCHEMA_VERSION, entries...)

expected_evaluation_updates(updates::Integer = HR_UPDATES) =
    collect(0:HR_EVALUATION_INTERVAL:Int(updates))

function corpus_split_seeds(study_config, split::Symbol)
    seeds = JLD2.jldopen(study_config.state_corpus, "r") do file
        corpus = read(file, "corpus")
        values = haskey(corpus, split) ? corpus[split] : corpus[string(split)]
        return sort!(Int.(collect(keys(values))))
    end
    expected = Dict(:train => 20, :validation => 1, :test => 2)[split]
    length(seeds) == expected || error(
        "Expected $expected $(study_config.label) $split bases, found $(length(seeds)).",
    )
    return seeds
end

function expected_corpus_files(study_value)
    study_config = study(study_value)
    files = String[]
    for split in (:train, :validation, :test)
        for base_seed in corpus_split_seeds(study_config, split), mirror in (false, true)
            push!(files, joinpath(
                study_config.distillation_root,
                "varying",
                string(split),
                "base_$(base_seed)_mirror_$(mirror ? 1 : 0).jld2",
            ))
        end
    end
    return files
end

function validate_baseline(path, study_config, expected_policy, run_file_sha, state_corpus_sha)
    JLD2.jldopen(path, "r") do file
        string(read(file, "status")) == "complete" || error("Incomplete baseline: $path")
        Float64(read(file, "rayleigh")) == study_config.rayleigh || error(
            "Baseline has the wrong Rayleigh number: $path",
        )
        Symbol(read(file, "protocol")) == study_config.protocol || error(
            "Baseline has the wrong protocol: $path",
        )
        Symbol(read(file, "policy")) == expected_policy || error(
            "Baseline has the wrong policy: $path",
        )
        Int(read(file, "steps")) == 200 || error("Baseline does not contain 200 steps: $path")
        Int(read(file, "case_count")) == 8 || error("Baseline does not contain eight cases: $path")
        string(read(file, "run_file_sha256")) == run_file_sha || error(
            "Baseline run-file identity is stale: $path",
        )
        string(read(file, "test_data_sha256")) == state_corpus_sha || error(
            "Baseline test-corpus identity is stale: $path",
        )
    end
    return nothing
end

function source_manifest(study_value)
    study_config = study(study_value)
    sources = (
        run_file = study_config.run_file,
        state_corpus = study_config.state_corpus,
        expert = study_config.expert,
        expert_baseline = study_config.expert_baseline,
        unactuated_baseline = study_config.unactuated_baseline,
    )
    for (name, path) in pairs(sources)
        isfile(path) || error("Missing $(study_config.label) source $name: $path")
    end
    corpus_files = expected_corpus_files(study_config.tag)
    length(corpus_files) == 46 || error(
        "Expected 46 $(study_config.label) distillation shards, found $(length(corpus_files)) planned paths.",
    )
    missing = filter(path -> !isfile(path), corpus_files)
    isempty(missing) || error(
        "Missing $(length(missing)) $(study_config.label) distillation shards; first: $(first(missing))",
    )
    run_file_sha = file_sha256(study_config.run_file)
    state_corpus_sha = file_sha256(study_config.state_corpus)
    expert_sha = file_sha256(study_config.expert)
    validate_baseline(
        study_config.expert_baseline,
        study_config,
        :deterministic_mean_action,
        run_file_sha,
        state_corpus_sha,
    )
    validate_baseline(
        study_config.unactuated_baseline,
        study_config,
        :zero_action,
        run_file_sha,
        state_corpus_sha,
    )
    JLD2.jldopen(study_config.expert_baseline, "r") do file
        string(read(file, "expert_sha256")) == expert_sha || error(
            "Expert baseline uses a different expert: $(study_config.expert_baseline)",
        )
    end
    return (;
        study = study_config.tag,
        protocol = study_config.protocol,
        rayleigh = study_config.rayleigh,
        run_file_path = abspath(study_config.run_file),
        run_file_sha256 = run_file_sha,
        state_corpus_path = abspath(study_config.state_corpus),
        state_corpus_sha256 = state_corpus_sha,
        expert_path = abspath(study_config.expert),
        expert_sha256 = expert_sha,
        expert_baseline_path = abspath(study_config.expert_baseline),
        expert_baseline_sha256 = file_sha256(study_config.expert_baseline),
        unactuated_baseline_path = abspath(study_config.unactuated_baseline),
        unactuated_baseline_sha256 = file_sha256(study_config.unactuated_baseline),
        distillation_root = abspath(study_config.distillation_root),
        distillation_shards = abspath.(corpus_files),
    )
end

end
