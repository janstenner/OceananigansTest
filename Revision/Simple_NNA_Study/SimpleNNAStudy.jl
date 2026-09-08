module SimpleNNAStudy

using Dates
using JLD2
using Printf
using SHA
using StableRNGs

export SNN_SCHEMA_VERSION, SNN_MASTER_SEED, SNN_UPDATES, SNN_BATCH_SIZE,
       SNN_VALIDATION_BATCH_SIZE, SNN_LEARNING_RATE, SNN_EVALUATION_INTERVAL,
       SNN_RESUME_INTERVAL, SNN_GARBAGE_COLLECTION_INTERVAL, SNN_REPLICATES,
       SNN_THRESHOLDS, SNN_QUALITY_THRESHOLD, SNN_CONFIGURATION_NAMES, SNN_STRENGTH_GRIDS,
       SNN_APPRENTICE_ARCHITECTURE, SNN_NNA_SCALE, SNN_HIDDEN_SIZE,
       SNN_MAT_ACTOR_PARAMETER_COUNT, SNN_PARAMETER_COUNT,
       configuration, normalize_configuration, normalize_experiment_id, seed_plan, seed_plan_hash,
       selected_variants, resolved_thresholds, study_jobs, job_for, run_directory, analysis_directory,
       status_path, analysis_status_path, atomic_save, load_status, write_status!,
       canonical_string, fingerprint, strength_tag, expected_evaluation_updates

const SNN_SCHEMA_VERSION = 1
const SNN_MASTER_SEED = 20_260_851
const SNN_UPDATES = 100_000
const SNN_BATCH_SIZE = 100
const SNN_VALIDATION_BATCH_SIZE = 512
const SNN_LEARNING_RATE = 2e-4
const SNN_EVALUATION_INTERVAL = 25
const SNN_RESUME_INTERVAL = 100
const SNN_GARBAGE_COLLECTION_INTERVAL = 5
const SNN_REPLICATES = 1:3
const SNN_THRESHOLDS = (0.0, 0.003, 0.006, 0.012)
const SNN_QUALITY_THRESHOLD = 2e-2
const SNN_APPRENTICE_ARCHITECTURE = :simple_nna
const SNN_NNA_SCALE = 10.2
const SNN_HIDDEN_SIZE = 102
const SNN_MAT_ACTOR_PARAMETER_COUNT = 47_698
const SNN_PARAMETER_COUNT = 47_432

function resolved_thresholds(values = Float64[])
    isempty(values) && return collect(SNN_THRESHOLDS)
    custom = sort!(unique(Float64.(values)))
    all(value -> isfinite(value) && value > 0, custom) || throw(ArgumentError(
        "Custom mask thresholds must be finite and positive; native threshold 0.0 is automatic.",
    ))
    return vcat(0.0, custom)
end

const SNN_CONFIGURATION_NAMES = (
    "go-gc", "go-sc", "gr-gc", "gr-sc",
)

const SNN_STRENGTH_GRIDS = Dict(
    "go-gc" => (0.008, 0.02, 0.05),                    # inherited Varying default: 0.025
    "go-sc" => (0.016, 0.04, 0.1),                     # inherited Varying default: 0.025
    "gr-gc" => (0.000024, 0.00006, 0.00015),           # inherited Varying default: 0.0001
    "gr-sc" => (0.000048, 0.00012, 0.0003),            # inherited Varying default: 0.0001
)

const SNN_CONFIGURATIONS = Dict(
    "go-gc" => (method = :go, grouping = :grouped_channels, group_channels = true),
    "go-sc" => (method = :go, grouping = :separate_channels, group_channels = false),
    "gr-gc" => (method = :gr, grouping = :grouped_channels, group_channels = true),
    "gr-sc" => (method = :gr, grouping = :separate_channels, group_channels = false),
)

function normalize_configuration(value)::String
    name = lowercase(strip(string(value)))
    haskey(SNN_CONFIGURATIONS, name) || throw(ArgumentError(
        "Unknown Simple-NNA configuration '$value'. Available: $(join(SNN_CONFIGURATION_NAMES, ", ")).",
    ))
    return name
end

configuration(value) = SNN_CONFIGURATIONS[normalize_configuration(value)]

function normalize_experiment_id(value)::String
    identifier = strip(string(value))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_-]*$", identifier) || throw(ArgumentError(
        "Experiment ID '$value' must contain only letters, digits, underscores, and hyphens.",
    ))
    return identifier
end

function seed_plan(replicate::Integer)
    replicate in SNN_REPLICATES || throw(ArgumentError("Replicate must be in 1:3."))
    planner = StableRNG(SNN_MASTER_SEED)
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

function strength_tag(strength::Real)
    value = Float64(strength)
    isfinite(value) && value > 0 || throw(ArgumentError("Strength must be finite and positive."))
    canonical = lowercase(@sprintf("%.12g", value))
    safe = replace(canonical, "." => "p", "+" => "", "-" => "m")
    return "s_" * safe
end

function selected_variants(selection = "all", strengths = Float64[])
    if lowercase(string(selection)) == "all"
        isempty(strengths) || throw(ArgumentError("Explicit strengths require exactly one --config."))
        return [(name = name, strength = strength) for name in SNN_CONFIGURATION_NAMES for strength in SNN_STRENGTH_GRIDS[name]]
    end
    name = normalize_configuration(selection)
    values = isempty(strengths) ? collect(SNN_STRENGTH_GRIDS[name]) : unique(Float64.(strengths))
    all(value -> isfinite(value) && value > 0, values) || throw(ArgumentError("Strengths must be finite and positive."))
    return [(name, strength = value) for value in values]
end

function job_for(experiment_id, configuration_name, strength::Real, replicate::Integer; updates::Integer = SNN_UPDATES)
    experiment = normalize_experiment_id(experiment_id)
    name = normalize_configuration(configuration_name)
    config = configuration(name)
    seeds = seed_plan(replicate)
    tag = strength_tag(strength)
    replicate_tag = @sprintf("r%02d", replicate)
    relative_path = joinpath(experiment, name, tag, replicate_tag)
    return (
        experiment_id = experiment,
        configuration = name,
        config...,
        regularization_strength = Float64(strength),
        strength_tag = tag,
        replicate = Int(replicate),
        seeds...,
        pairing_hash = seed_plan_hash(replicate),
        updates = Int(updates),
        id = "snn_$(experiment)_$(replace(name, "-" => "_"))_$(tag)_$(replicate_tag)",
        relative_path,
    )
end

function study_jobs(experiment_id, selection = "all", strengths = Float64[]; updates::Integer = SNN_UPDATES)
    return [
        job_for(experiment_id, variant.name, variant.strength, replicate; updates)
        for variant in selected_variants(selection, strengths)
        for replicate in SNN_REPLICATES
    ]
end

run_directory(results_root::AbstractString, job) = joinpath(abspath(results_root), job.relative_path)
analysis_directory(results_root::AbstractString, experiment_id, configuration_name) =
    joinpath(abspath(results_root), normalize_experiment_id(experiment_id), normalize_configuration(configuration_name), "analysis")
status_path(results_root::AbstractString, job) = joinpath(run_directory(results_root, job), "status.jld2")
analysis_status_path(results_root::AbstractString, experiment_id, configuration_name) =
    joinpath(analysis_directory(results_root, experiment_id, configuration_name), "status.jld2")

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

function write_status!(path::AbstractString; entries...)
    return atomic_save(path; schema_version = SNN_SCHEMA_VERSION, entries...)
end

expected_evaluation_updates(updates::Integer = SNN_UPDATES) = collect(0:SNN_EVALUATION_INTERVAL:Int(updates))

end
