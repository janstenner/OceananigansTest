module Package6Study

using Dates
using JLD2
using Printf
using SHA
using StableRNGs

export P6_SCHEMA_VERSION, P6_MASTER_SEED, P6_STRENGTHS, P6_GR_STRENGTH,
       P6_UPDATES, P6_EVALUATION_INTERVAL, P6_RESUME_INTERVAL,
       P6_GARBAGE_COLLECTION_INTERVAL, P6_REGRESSION_LEARNING_RATE,
       P6_TRAINING_BATCH_SIZE, P6_VALIDATION_BATCH_SIZE,
       P6_REPLICATES, P6_POLL_SECONDS, P6_TIMEOUT_SECONDS,
       normalize_protocol, seed_plan, seed_plan_hash, study_jobs, analysis_jobs,
       job_for, run_id, run_relative_path, run_directory, status_path,
       canonical_string, fingerprint, atomic_save, load_status, write_status!,
       expected_evaluation_updates, short_path_components

const P6_SCHEMA_VERSION = 1
const P6_MASTER_SEED = 20_260_810
const P6_STRENGTHS = (0.0015, 0.003, 0.006, 0.01, 0.03)
const P6_GR_STRENGTH = Dict(:fixed => 0.00004, :varying => 0.0001)
const P6_UPDATES = Dict(:fixed => 35_000, :varying => 50_000)
const P6_EVALUATION_INTERVAL = 25
const P6_RESUME_INTERVAL = 100
const P6_GARBAGE_COLLECTION_INTERVAL = 5
const P6_REGRESSION_LEARNING_RATE = 2e-4
const P6_TRAINING_BATCH_SIZE = Dict(:fixed => 50, :varying => 100)
const P6_VALIDATION_BATCH_SIZE = Dict(:fixed => 200, :varying => 512)
const P6_REPLICATES = 1:3
const P6_POLL_SECONDS = 60
const P6_TIMEOUT_SECONDS = 14 * 24 * 60 * 60

function normalize_protocol(value)::Symbol
    protocol = Symbol(lowercase(string(value)))
    protocol in (:fixed, :varying) || throw(
        ArgumentError("Protocol must be fixed or varying, got '$value'."),
    )
    return protocol
end

"""Return the deterministic, paired apprentice and batch seeds for one replicate."""
function seed_plan(replicate::Integer)
    replicate in P6_REPLICATES || throw(
        ArgumentError("Replicate must be in $(first(P6_REPLICATES)):$(last(P6_REPLICATES))."),
    )
    planner = StableRNG(P6_MASTER_SEED)
    apprentice_seed = 0
    batch_seed = 0
    for _ in 1:replicate
        apprentice_seed = rand(planner, 1:2_000_000_000)
        batch_seed = rand(planner, 1:2_000_000_000)
    end
    apprentice_seed == 600_601 && error("The calibration seed must not be reused.")
    batch_seed == 600_601 && error("The calibration seed must not be reused.")
    return (; replicate = Int(replicate), apprentice_seed, batch_seed)
end

function canonical_string(value)
    if value isa AbstractDict
        entries = sort!(collect(pairs(value)); by = pair -> string(first(pair)))
        return "{" * join([
            canonical_string(first(entry)) * ":" * canonical_string(last(entry))
            for entry in entries
        ], ",") * "}"
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

function seed_plan_hash(replicate::Integer)
    return fingerprint(seed_plan(replicate))
end

method_tag(method::Symbol) = method === :go ? "go" : method === :gr ? "gr" : error("Unknown method $method.")
protocol_tag(protocol::Symbol) = normalize_protocol(protocol) === :fixed ? "f" : "v"
replicate_tag(replicate::Integer) = @sprintf("r%02d", replicate)
strength_tag(index::Integer) = @sprintf("s%02d", index)

function run_id(protocol::Symbol, method::Symbol, strength_index::Integer, replicate::Integer)
    suffix = method === :go ? "_$(strength_tag(strength_index))" : ""
    return "p6_$(protocol_tag(protocol))_$(method_tag(method))$(suffix)_$(replicate_tag(replicate))"
end

function run_relative_path(protocol::Symbol, method::Symbol, strength_index::Integer, replicate::Integer)
    pieces = String[string(normalize_protocol(protocol)), method_tag(method)]
    method === :go && push!(pieces, strength_tag(strength_index))
    push!(pieces, replicate_tag(replicate))
    return joinpath(pieces...)
end

run_directory(results_root::AbstractString, job) = joinpath(abspath(results_root), job.relative_path)
status_path(results_root::AbstractString, job) = joinpath(run_directory(results_root, job), "status.jld2")

function study_jobs(protocol_selection = :all)
    protocols = protocol_selection === :all ? (:fixed, :varying) : (normalize_protocol(protocol_selection),)
    jobs = NamedTuple[]
    for protocol in protocols
        for (strength_index, strength) in enumerate(P6_STRENGTHS), replicate in P6_REPLICATES
            seeds = seed_plan(replicate)
            push!(jobs, (
                protocol,
                method = :go,
                strength_index,
                regularization_strength = Float64(strength),
                replicate = Int(replicate),
                seeds...,
                pairing_hash = seed_plan_hash(replicate),
                updates = P6_UPDATES[protocol],
                id = run_id(protocol, :go, strength_index, replicate),
                relative_path = run_relative_path(protocol, :go, strength_index, replicate),
            ))
        end
        for replicate in P6_REPLICATES
            seeds = seed_plan(replicate)
            push!(jobs, (
                protocol,
                method = :gr,
                strength_index = 0,
                regularization_strength = P6_GR_STRENGTH[protocol],
                replicate = Int(replicate),
                seeds...,
                pairing_hash = seed_plan_hash(replicate),
                updates = P6_UPDATES[protocol],
                id = run_id(protocol, :gr, 0, replicate),
                relative_path = run_relative_path(protocol, :gr, 0, replicate),
            ))
        end
    end
    return jobs
end

function analysis_jobs(protocol_selection = :all)
    protocols = protocol_selection === :all ? (:fixed, :varying) : (normalize_protocol(protocol_selection),)
    return [(
        protocol,
        id = "p6_$(protocol_tag(protocol))_analyze",
        relative_path = joinpath(string(protocol), "analysis"),
    ) for protocol in protocols]
end

function job_for(protocol, method, strength_index, replicate)
    protocol = normalize_protocol(protocol)
    method = Symbol(lowercase(string(method)))
    matches = filter(study_jobs(protocol)) do job
        job.method === method && job.strength_index == Int(strength_index) && job.replicate == Int(replicate)
    end
    length(matches) == 1 || error("No unique Package-6 job matches the requested identity.")
    return only(matches)
end

expected_evaluation_updates(updates::Integer) = collect(0:P6_EVALUATION_INTERVAL:Int(updates))

function short_path_components(job; maximum::Int = 16)
    components = splitpath(job.relative_path)
    return all(component -> ncodeunits(component) <= maximum, components)
end

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
    return path
end

function load_status(path::AbstractString)
    isfile(path) || return nothing
    loaded = JLD2.load(path)
    Int(loaded["schema_version"]) == P6_SCHEMA_VERSION || error("Unsupported status schema at '$path'.")
    return Dict{Symbol, Any}(Symbol(key) => value for (key, value) in loaded)
end

function write_status!(path::AbstractString; state::Symbol, entries...)
    state in (:running, :complete, :failed) || error("Unsupported worker state '$state'.")
    return atomic_save(
        path;
        schema_version = P6_SCHEMA_VERSION,
        state,
        updated_at = string(Dates.now()),
        entries...,
    )
end

end
