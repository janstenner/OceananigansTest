using Dates
using JLD2
using Printf
using SHA

const PARETO_ARCHIVE_SCHEMA_VERSION = 1

Base.@kwdef struct CandidateSchedule
    start_update::Int = 0
    evaluation_interval::Int = 100
    garbage_collection_interval::Int = 10
    resume_interval::Int = 100

    function CandidateSchedule(
        start_update::Int,
        evaluation_interval::Int,
        garbage_collection_interval::Int,
        resume_interval::Int,
    )
        start_update >= 0 || throw(ArgumentError("start_update must be nonnegative."))
        evaluation_interval > 0 || throw(ArgumentError("evaluation_interval must be positive."))
        garbage_collection_interval > 0 || throw(ArgumentError("garbage_collection_interval must be positive."))
        resume_interval > 0 || throw(ArgumentError("resume_interval must be positive."))
        return new(
            start_update,
            evaluation_interval,
            garbage_collection_interval,
            resume_interval,
        )
    end
end

function candidate_schedule_dict(schedule::CandidateSchedule)
    return Dict{Symbol, Any}(
        :start_update => schedule.start_update,
        :evaluation_interval => schedule.evaluation_interval,
        :garbage_collection_interval => schedule.garbage_collection_interval,
        :resume_interval => schedule.resume_interval,
    )
end

function should_evaluate_candidates(
    schedule::CandidateSchedule,
    update::Integer;
    final::Bool = false,
)
    update >= schedule.start_update || return false
    final && return true
    return mod(Int(update) - schedule.start_update, schedule.evaluation_interval) == 0
end

function should_save_resume(schedule::CandidateSchedule, update::Integer; final::Bool = false)
    final && return true
    return mod(Int(update), schedule.resume_interval) == 0
end

function canonical_config_string(value)
    if value isa AbstractDict
        entries = sort!(collect(pairs(value)); by = pair -> string(first(pair)))
        return "{" * join(
            [
                canonical_config_string(first(entry)) * ":" *
                canonical_config_string(last(entry))
                for entry in entries
            ],
            ",",
        ) * "}"
    elseif value isa NamedTuple
        return canonical_config_string(Dict(pairs(value)))
    elseif value isa Tuple || value isa AbstractVector
        return "[" * join(canonical_config_string.(collect(value)), ",") * "]"
    elseif value isa Symbol
        return ":" * string(value)
    elseif value isa AbstractString
        return repr(String(value))
    elseif value === nothing
        return "nothing"
    end
    return repr(value)
end

function pareto_config_fingerprint(config)
    return bytes2hex(SHA.sha256(codeunits(canonical_config_string(config))))
end

function pareto_atomic_save(path::AbstractString; entries...)
    mkpath(dirname(path))
    temporary_path = path * ".tmp.$(getpid()).$(time_ns())"
    try
        JLD2.jldopen(temporary_path, "w") do file
            for (key, value) in pairs(entries)
                file[string(key)] = value
            end
        end
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return path
end

function archive_value(container, key::Symbol; default = missing)
    if haskey(container, key)
        return container[key]
    elseif haskey(container, string(key))
        return container[string(key)]
    elseif default !== missing
        return default
    end
    error("Archive data has no '$key' entry.")
end

mutable struct ParetoArchiveManager
    run_directory::String
    run_id::String
    schedule::CandidateSchedule
    config::Dict{Symbol, Any}
    config_fingerprint::String
    front::Vector{Dict{Symbol, Any}}
    evaluation_count::Int
    last_evaluated_update::Int
end

function archive_directory(manager::ParetoArchiveManager)
    return joinpath(manager.run_directory, "archive")
end

function evaluation_directory(manager::ParetoArchiveManager)
    return joinpath(manager.run_directory, "evaluations")
end

function candidate_directory(manager::ParetoArchiveManager)
    return joinpath(manager.run_directory, "candidates")
end

function resume_directory(manager::ParetoArchiveManager)
    return joinpath(manager.run_directory, "resume")
end

function archive_manifest_path(manager::ParetoArchiveManager)
    return joinpath(archive_directory(manager), "archive.jld2")
end

function run_config_path(manager::ParetoArchiveManager)
    return joinpath(manager.run_directory, "config.jld2")
end

function evaluation_path(manager::ParetoArchiveManager, update::Integer)
    return joinpath(evaluation_directory(manager), @sprintf("update_%012d.jld2", update))
end

function candidate_checkpoint_path(manager::ParetoArchiveManager, update::Integer)
    return joinpath(candidate_directory(manager), @sprintf("checkpoint_%012d.jld2", update))
end

function latest_resume_path(manager::ParetoArchiveManager)
    return joinpath(resume_directory(manager), "latest.jld2")
end

function normalize_archive_dict(raw)
    return Dict{Symbol, Any}(
        (key isa Symbol ? key : Symbol(key)) => value for (key, value) in raw
    )
end

function candidate_identifier(run_id, update, threshold_id)
    material = "$run_id|$(Int(update))|$(string(threshold_id))"
    return bytes2hex(SHA.sha256(codeunits(material)))[1:24]
end

function normalize_candidate_record(candidate, run_id::String, update::Integer)
    record = normalize_archive_dict(candidate)
    for key in (:threshold_id, :validation_matching, :active_inputs, :mask, :numeric_status)
        haskey(record, key) || error("Candidate is missing required field :$key.")
    end
    record[:run_id] = run_id
    record[:update] = Int(update)
    record[:checkpoint_id] = @sprintf("checkpoint_%012d", update)
    record[:candidate_id] = candidate_identifier(run_id, update, record[:threshold_id])
    record[:validation_matching] = Float64(record[:validation_matching])
    record[:active_inputs] = Int(record[:active_inputs])
    record[:numeric_status] = Symbol(record[:numeric_status])
    record[:model_path] = nothing
    record[:loadable] = false
    return record
end

function valid_pareto_candidate(candidate)
    return archive_value(candidate, :numeric_status) === :ok &&
           isfinite(Float64(archive_value(candidate, :validation_matching))) &&
           Int(archive_value(candidate, :active_inputs)) >= 0
end

function candidate_dominates(left, right)
    valid_pareto_candidate(left) || return false
    valid_pareto_candidate(right) || return true
    left_inputs = Int(archive_value(left, :active_inputs))
    right_inputs = Int(archive_value(right, :active_inputs))
    left_matching = Float64(archive_value(left, :validation_matching))
    right_matching = Float64(archive_value(right, :validation_matching))
    return left_inputs <= right_inputs &&
           left_matching <= right_matching &&
           (left_inputs < right_inputs || left_matching < right_matching)
end

function equivalent_objectives(left, right)
    return Int(archive_value(left, :active_inputs)) == Int(archive_value(right, :active_inputs)) &&
           Float64(archive_value(left, :validation_matching)) == Float64(archive_value(right, :validation_matching))
end

function pareto_front(candidates)
    valid = [normalize_archive_dict(candidate) for candidate in candidates if valid_pareto_candidate(candidate)]
    sort!(
        valid;
        by = candidate -> (
            Int(archive_value(candidate, :active_inputs)),
            Float64(archive_value(candidate, :validation_matching)),
            string(archive_value(candidate, :candidate_id)),
        ),
    )
    result = Dict{Symbol, Any}[]
    for candidate in valid
        if any(existing -> candidate_dominates(existing, candidate), result)
            continue
        end
        if any(existing -> equivalent_objectives(existing, candidate), result)
            continue
        end
        filter!(existing -> !candidate_dominates(candidate, existing), result)
        push!(result, candidate)
    end
    sort!(
        result;
        by = candidate -> (
            Int(archive_value(candidate, :active_inputs)),
            Float64(archive_value(candidate, :validation_matching)),
        ),
    )
    return result
end

function save_archive_manifest!(manager::ParetoArchiveManager)
    return pareto_atomic_save(
        archive_manifest_path(manager);
        schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
        run_id = manager.run_id,
        config_fingerprint = manager.config_fingerprint,
        schedule = candidate_schedule_dict(manager.schedule),
        front = manager.front,
        evaluation_count = manager.evaluation_count,
        last_evaluated_update = manager.last_evaluated_update,
        updated_at = string(Dates.now()),
    )
end

function evaluation_files(manager::ParetoArchiveManager)
    directory = evaluation_directory(manager)
    isdir(directory) || return String[]
    return sort!(
        [
            joinpath(directory, filename)
            for filename in readdir(directory)
            if startswith(filename, "update_") && endswith(filename, ".jld2")
        ],
    )
end

function load_evaluation_records(path::AbstractString)
    loaded = JLD2.load(path)
    Int(loaded["schema_version"]) == PARETO_ARCHIVE_SCHEMA_VERSION || error(
        "Unsupported evaluation schema in '$path'.",
    )
    return [normalize_archive_dict(record) for record in loaded["candidates"]]
end

function rebuild_pareto_archive!(manager::ParetoArchiveManager)
    records = Dict{Symbol, Any}[]
    last_update = -1
    files = evaluation_files(manager)
    for path in files
        loaded = JLD2.load(path)
        loaded["config_fingerprint"] == manager.config_fingerprint || error(
            "Evaluation '$path' belongs to a different experiment configuration.",
        )
        append!(records, [normalize_archive_dict(record) for record in loaded["candidates"]])
        last_update = max(last_update, Int(loaded["update"]))
    end
    manager.front = pareto_front(records)
    manager.evaluation_count = length(files)
    manager.last_evaluated_update = last_update
    for candidate in manager.front
        model_path = archive_value(candidate, :model_path; default = nothing)
        !isnothing(model_path) && isfile(model_path) || error(
            "Rebuilt Pareto candidate $(candidate[:candidate_id]) has no loadable model.",
        )
    end
    save_archive_manifest!(manager)
    return manager
end


function initialize_pareto_archive(
    run_directory::AbstractString;
    run_id,
    schedule::CandidateSchedule,
    config = Dict{Symbol, Any}(),
)
    config_dict = Dict{Symbol, Any}(normalize_archive_dict(config))
    fingerprint_payload = Dict{Symbol, Any}(
        :run_id => string(run_id),
        :schedule => candidate_schedule_dict(schedule),
        :config => config_dict,
    )
    fingerprint = pareto_config_fingerprint(fingerprint_payload)
    manager = ParetoArchiveManager(
        abspath(run_directory),
        string(run_id),
        schedule,
        config_dict,
        fingerprint,
        Dict{Symbol, Any}[],
        0,
        -1,
    )
    for directory in (
        manager.run_directory,
        archive_directory(manager),
        evaluation_directory(manager),
        candidate_directory(manager),
        resume_directory(manager),
    )
        mkpath(directory)
    end

    config_path = run_config_path(manager)
    if isfile(config_path)
        loaded = JLD2.load(config_path)
        loaded["config_fingerprint"] == fingerprint || error(
            "Run directory already contains a different experiment configuration: $(manager.run_directory)",
        )
    else
        pareto_atomic_save(
            config_path;
            schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
            run_id = manager.run_id,
            config = manager.config,
            schedule = candidate_schedule_dict(schedule),
            config_fingerprint = fingerprint,
            created_at = string(Dates.now()),
        )
    end

    files = evaluation_files(manager)
    if !isempty(files)
        # Evaluation shards are the durable scientific event log. Rebuilding
        # from them also recovers a crash after writing an evaluation but
        # before publishing the corresponding manifest.
        rebuild_pareto_archive!(manager)
    elseif isfile(archive_manifest_path(manager))
        loaded = JLD2.load(archive_manifest_path(manager))
        loaded["config_fingerprint"] == fingerprint || error(
            "Archive manifest belongs to a different experiment configuration.",
        )
        manager.front = [normalize_archive_dict(record) for record in loaded["front"]]
        manager.evaluation_count = Int(loaded["evaluation_count"])
        manager.last_evaluated_update = Int(loaded["last_evaluated_update"])
    else
        save_archive_manifest!(manager)
    end
    garbage_collect_candidate_models!(manager)
    return manager
end

function save_candidate_checkpoint!(manager::ParetoArchiveManager, update::Integer, model_payload)
    path = candidate_checkpoint_path(manager, update)
    isfile(path) && return path
    pareto_atomic_save(
        path;
        schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
        run_id = manager.run_id,
        update = Int(update),
        model_payload,
        created_at = string(Dates.now()),
    )
    return path
end

function record_candidate_batch!(
    manager::ParetoArchiveManager,
    update::Integer,
    candidates;
    model_payload,
    evaluation_metadata = Dict{Symbol, Any}(),
)
    update = Int(update)
    update >= manager.schedule.start_update || throw(
        ArgumentError("Update $update is before candidate start $(manager.schedule.start_update)."),
    )
    path = evaluation_path(manager, update)
    if isfile(path)
        loaded = JLD2.load(path)
        loaded["config_fingerprint"] == manager.config_fingerprint || error(
            "Existing evaluation at update $update has a different configuration.",
        )
        return [normalize_archive_dict(record) for record in loaded["candidates"]]
    end

    records = [normalize_candidate_record(candidate, manager.run_id, update) for candidate in candidates]
    candidate_ids = string.(getindex.(records, :candidate_id))
    length(unique(candidate_ids)) == length(candidate_ids) || error(
        "Threshold IDs must be unique within update $update.",
    )

    proposed_front = pareto_front(vcat(manager.front, records))
    proposed_ids = Set(string(candidate[:candidate_id]) for candidate in proposed_front)
    new_survivor_ids = Set(
        string(record[:candidate_id]) for record in records if string(record[:candidate_id]) in proposed_ids
    )
    checkpoint_path = nothing
    if !isempty(new_survivor_ids)
        checkpoint_path = save_candidate_checkpoint!(manager, update, model_payload)
        for record in records
            if string(record[:candidate_id]) in new_survivor_ids
                record[:model_path] = checkpoint_path
                record[:loadable] = true
            end
        end
        updated_by_id = Dict(string(record[:candidate_id]) => record for record in records)
        for index in eachindex(proposed_front)
            candidate_id = string(proposed_front[index][:candidate_id])
            haskey(updated_by_id, candidate_id) && (proposed_front[index] = updated_by_id[candidate_id])
        end
    end

    pareto_atomic_save(
        path;
        schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
        run_id = manager.run_id,
        update,
        candidates = records,
        evaluation_metadata = normalize_archive_dict(evaluation_metadata),
        config_fingerprint = manager.config_fingerprint,
        created_at = string(Dates.now()),
    )

    manager.front = proposed_front
    manager.evaluation_count += 1
    manager.last_evaluated_update = max(manager.last_evaluated_update, update)
    save_archive_manifest!(manager)

    if mod(manager.evaluation_count, manager.schedule.garbage_collection_interval) == 0
        garbage_collect_candidate_models!(manager)
    end
    return records
end

function referenced_candidate_models(manager::ParetoArchiveManager)
    return Set(
        abspath(string(candidate[:model_path]))
        for candidate in manager.front
        if get(candidate, :loadable, false) && !isnothing(get(candidate, :model_path, nothing))
    )
end

function garbage_collect_candidate_models!(manager::ParetoArchiveManager)
    referenced = referenced_candidate_models(manager)
    directory = candidate_directory(manager)
    deleted = String[]
    isdir(directory) || return deleted
    for filename in readdir(directory)
        endswith(filename, ".jld2") || continue
        path = abspath(joinpath(directory, filename))
        abspath(dirname(path)) == abspath(directory) || error("Unsafe candidate cleanup path '$path'.")
        if !(path in referenced)
            rm(path; force = true)
            push!(deleted, path)
        end
    end
    return deleted
end

function save_resume_checkpoint!(
    manager::ParetoArchiveManager,
    update::Integer,
    resume_state;
    status::Symbol = :running,
)
    return pareto_atomic_save(
        latest_resume_path(manager);
        schema_version = PARETO_ARCHIVE_SCHEMA_VERSION,
        run_id = manager.run_id,
        update = Int(update),
        status,
        resume_state,
        config_fingerprint = manager.config_fingerprint,
        saved_at = string(Dates.now()),
    )
end

function load_resume_checkpoint(manager::ParetoArchiveManager)
    path = latest_resume_path(manager)
    isfile(path) || return nothing
    loaded = JLD2.load(path)
    loaded["config_fingerprint"] == manager.config_fingerprint || error(
        "Resume checkpoint belongs to a different experiment configuration.",
    )
    return (
        update = Int(loaded["update"]),
        status = Symbol(loaded["status"]),
        resume_state = loaded["resume_state"],
        path,
    )
end

function finalize_pareto_archive!(manager::ParetoArchiveManager)
    deleted = garbage_collect_candidate_models!(manager)
    save_archive_manifest!(manager)
    return (front = manager.front, deleted)
end
