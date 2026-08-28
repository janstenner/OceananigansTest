module Package7Study

using Dates
using JLD2
using Printf
using SHA
using StableRNGs

export P7_SCHEMA_VERSION, P7_MASTER_SEED, P7_UPDATES, P7_BATCH_SIZE,
       P7_VALIDATION_BATCH_SIZE, P7_LEARNING_RATE, P7_EVALUATION_INTERVAL,
       P7_RESUME_INTERVAL, P7_GARBAGE_COLLECTION_INTERVAL, P7_REPLICATES,
       P7_THRESHOLDS, P7_CONFIGURATION_NAMES, P7_DEFAULT_STRENGTHS,
       configuration, normalize_configuration, seed_plan, seed_plan_hash,
       selected_variants, study_jobs, job_for, run_directory, analysis_directory,
       status_path, analysis_status_path, atomic_save, load_status, write_status!,
       canonical_string, fingerprint, strength_tag, expected_evaluation_updates

const P7_SCHEMA_VERSION = 1
const P7_MASTER_SEED = 20_260_829
const P7_UPDATES = 35_000
const P7_BATCH_SIZE = 50
const P7_VALIDATION_BATCH_SIZE = 200
const P7_LEARNING_RATE = 2e-4
const P7_EVALUATION_INTERVAL = 25
const P7_RESUME_INTERVAL = 100
const P7_GARBAGE_COLLECTION_INTERVAL = 5
const P7_REPLICATES = 1:3
const P7_THRESHOLDS = (0.0, 0.0015, 0.003, 0.005)

const P7_CONFIGURATION_NAMES = (
    "go-gc", "go-sc", "gr-gc", "gr-sc",
    "group-lasso-gc", "group-lasso-sc", "growl-gc", "growl-sc",
)

const P7_DEFAULT_STRENGTHS = Dict(
    "go-gc" => 0.09,
    "go-sc" => 0.09,
    "gr-gc" => 0.00004,
    "gr-sc" => 0.00004,
    "group-lasso-gc" => 0.0001,
    "group-lasso-sc" => 0.0001,
    "growl-gc" => 0.00006,
    "growl-sc" => 0.00006,
)

const P7_CONFIGURATIONS = Dict(
    "go-gc" => (method = :go, grouping = :grouped_channels, group_channels = true),
    "go-sc" => (method = :go, grouping = :separate_channels, group_channels = false),
    "gr-gc" => (method = :gr, grouping = :grouped_channels, group_channels = true),
    "gr-sc" => (method = :gr, grouping = :separate_channels, group_channels = false),
    "group-lasso-gc" => (method = :group_lasso, grouping = :grouped_channels, group_channels = true),
    "group-lasso-sc" => (method = :group_lasso, grouping = :separate_channels, group_channels = false),
    "growl-gc" => (method = :growl, grouping = :grouped_channels, group_channels = true),
    "growl-sc" => (method = :growl, grouping = :separate_channels, group_channels = false),
)

function normalize_configuration(value)::String
    name = lowercase(strip(string(value)))
    haskey(P7_CONFIGURATIONS, name) || throw(ArgumentError(
        "Unknown Package-7 configuration '$value'. Available: $(join(P7_CONFIGURATION_NAMES, ", ")).",
    ))
    return name
end

configuration(value) = P7_CONFIGURATIONS[normalize_configuration(value)]

function seed_plan(replicate::Integer)
    replicate in P7_REPLICATES || throw(ArgumentError("Replicate must be in 1:3."))
    planner = StableRNG(P7_MASTER_SEED)
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
        return [(name = name, strength = P7_DEFAULT_STRENGTHS[name]) for name in P7_CONFIGURATION_NAMES]
    end
    name = normalize_configuration(selection)
    values = isempty(strengths) ? [P7_DEFAULT_STRENGTHS[name]] : unique(Float64.(strengths))
    all(value -> isfinite(value) && value > 0, values) || throw(ArgumentError("Strengths must be finite and positive."))
    return [(name, strength = value) for value in values]
end

function job_for(configuration_name, strength::Real, replicate::Integer; updates::Integer = P7_UPDATES)
    name = normalize_configuration(configuration_name)
    config = configuration(name)
    seeds = seed_plan(replicate)
    tag = strength_tag(strength)
    replicate_tag = @sprintf("r%02d", replicate)
    relative_path = joinpath(name, tag, replicate_tag)
    return (
        configuration = name,
        config...,
        regularization_strength = Float64(strength),
        strength_tag = tag,
        replicate = Int(replicate),
        seeds...,
        pairing_hash = seed_plan_hash(replicate),
        updates = Int(updates),
        id = "p7_$(replace(name, "-" => "_"))_$(tag)_$(replicate_tag)",
        relative_path,
    )
end

function study_jobs(selection = "all", strengths = Float64[]; updates::Integer = P7_UPDATES)
    return [
        job_for(variant.name, variant.strength, replicate; updates)
        for variant in selected_variants(selection, strengths)
        for replicate in P7_REPLICATES
    ]
end

run_directory(results_root::AbstractString, job) = joinpath(abspath(results_root), job.relative_path)
analysis_directory(results_root::AbstractString, configuration_name, strength::Real) =
    joinpath(abspath(results_root), normalize_configuration(configuration_name), strength_tag(strength), "analysis")
status_path(results_root::AbstractString, job) = joinpath(run_directory(results_root, job), "status.jld2")
analysis_status_path(results_root::AbstractString, configuration_name, strength::Real) =
    joinpath(analysis_directory(results_root, configuration_name, strength), "status.jld2")

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
    return atomic_save(path; schema_version = P7_SCHEMA_VERSION, entries...)
end

expected_evaluation_updates(updates::Integer = P7_UPDATES) = collect(0:P7_EVALUATION_INTERVAL:Int(updates))

end
