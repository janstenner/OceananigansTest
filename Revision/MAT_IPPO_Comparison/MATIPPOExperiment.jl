module MATIPPOExperiment

using Dates
using Flux
using JLD2
using Printf
using Random
using RL
using SHA
using Sockets
using StableRNGs
using Statistics

export DEFAULT_EPISODES, DEFAULT_RESULTS_DIRECTORY, import_package3!, prepare_jobs,
       run_worker, collect_plan_entries, result_path, validation_path

const SCHEMA_VERSION = 1
const DEFAULT_EPISODES = Dict(:fixed => 2_000, :varying => 4_000)
const ALGORITHMS = (:mat, :ippo)
const PROTOCOLS = (:fixed, :varying)
const EXPECTED_PARAMETERS = Dict(
    (:fixed, :mat) => 46_499,
    (:fixed, :ippo) => 46_931,
    (:varying, :mat) => 75_551,
    (:varying, :ippo) => 75_099,
)
const REVISION_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = normpath(joinpath(REVISION_DIRECTORY, ".."))
const RL_ROOT = normpath(joinpath(dirname(pathof(RL)), ".."))
const DEFAULT_RESULTS_DIRECTORY = joinpath(@__DIR__, "results")
const DEFAULT_PACKAGE3_DIRECTORY = joinpath(REVISION_DIRECTORY, "MAT_Stability", "results")
const CORPUS_PATH = joinpath(
    REVISION_DIRECTORY,
    "VaryingIC_Corpus",
    "varying_ic_corpus.jld2",
)
const MAT_CONFIG = (
    name = :modified_full,
    useSeparateValueChain = true,
    useLayerNorm = false,
    useSelfAttentionFirst = false,
    use_mus = true,
)
const SNAPSHOT_GLOBALS = (
    :Ra, :Pr, :Nx, :Nz, :Lx, :Lz, :inner_dt, :dt, :actuators, :sensors,
    :update_freq, :learning_rate, :n_epochs, :n_microbatches, :clip_range,
    :target_kl, :nna_scale, :nna_scale_critic, :dim_model, :block_num,
    :head_num, :head_dim, :ffn_dim, :drop_out, :betas, :positional_encoding,
    :positional_encoding_decoder, :customCrossAttention, :jointPPO,
    :one_by_one_training, :useSeparateValueChain, :useLayerNorm,
    :useSelfAttentionFirst, :use_mus, :randomIC, :joon_pe, :square_rewards,
)

# Populated by exactly one dynamically included Package-2 run file per process.
agent = nothing
hook = nothing
env = nothing
model = nothing
function initialize_setup end
function generate_random_init end

normalize_protocol(value) = begin
    result = Symbol(lowercase(string(value)))
    result in PROTOCOLS || throw(ArgumentError("Unknown protocol '$value'."))
    result
end

normalize_algorithm(value) = begin
    result = Symbol(lowercase(string(value)))
    result in ALGORITHMS || throw(ArgumentError("Unknown algorithm '$value'."))
    result
end

function selected_protocols(value)
    result = Symbol(lowercase(string(value)))
    result === :all && return collect(PROTOCOLS)
    return [normalize_protocol(result)]
end

source_hash(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))
run_id(run_seed::Integer, ic_seed::Integer) =
    "seed_" * bytes2hex(SHA.sha256("$(Int(run_seed)):$(Int(ic_seed))"))[1:16]

plan_path(results_directory::AbstractString) = joinpath(results_directory, "run_plan.jld2")
result_path(results_directory::AbstractString, id, protocol, algorithm) = joinpath(
    results_directory,
    "runs",
    string(id),
    string(normalize_protocol(protocol)),
    string(normalize_algorithm(algorithm)) * ".jld2",
)
validation_path(results_directory::AbstractString, id, protocol, algorithm) = replace(
    result_path(results_directory, id, protocol, algorithm),
    r"\.jld2$" => "_validation.jld2",
)

function empty_plan(master_seed::Integer = rand(RandomDevice(), 1:2_000_000_000))
    return Dict{String, Any}(
        "schema_version" => SCHEMA_VERSION,
        "master_seed" => Int(master_seed),
        "generated_pairs" => 0,
        "created_at" => string(now()),
        "updated_at" => string(now()),
        "entries" => Any[],
    )
end

function load_plan(results_directory::AbstractString; allow_missing::Bool = false)
    path = plan_path(results_directory)
    if !isfile(path)
        allow_missing && return empty_plan()
        error("No Package-4 plan found at '$path'.")
    end
    plan = JLD2.load(path, "plan")
    Int(plan["schema_version"]) == SCHEMA_VERSION || error("Unsupported plan schema.")
    return plan
end

function atomic_jldsave(path::AbstractString; values...)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    rm(temporary; force = true)
    JLD2.jldsave(temporary; values...)
    mv(temporary, path; force = true)
    return path
end

function save_plan(results_directory::AbstractString, plan)
    plan["updated_at"] = string(now())
    return atomic_jldsave(plan_path(results_directory); plan)
end

function acquire_lock(path::AbstractString)
    mkpath(dirname(path))
    try
        mkdir(path)
    catch error_value
        isdir(path) || rethrow(error_value)
        error(
            "Lock '$path' already exists. Another process may be active; " *
            "remove it only after checking running workers.",
        )
    end
    open(joinpath(path, "owner.txt"), "w") do io
        println(io, "pid=$(getpid())")
        println(io, "host=$(gethostname())")
        println(io, "started_at=$(now())")
    end
    return path
end

function with_plan_lock(f, results_directory::AbstractString)
    mkpath(results_directory)
    lock = acquire_lock(joinpath(results_directory, ".plan.lock"))
    try
        return f()
    finally
        rm(lock; recursive = true, force = true)
    end
end

function normalize_trace(raw)
    return [
        (
            split = Symbol(item.split),
            base_seed = Int(item.base_seed),
            mirror = Bool(item.mirror),
            offset = Int(item.offset),
        ) for item in raw
    ]
end

function training_seeds()
    raw = JLD2.load(CORPUS_PATH, "corpus")
    split = haskey(raw, :train) ? raw[:train] : raw["train"]
    return sort!(Int.(collect(keys(split))))
end

function varying_schedule(ic_seed::Integer, episode_count::Integer)
    seeds = training_seeds()
    rng = StableRNG(Int(ic_seed))
    return [
        (
            split = :train,
            base_seed = rand(rng, seeds),
            mirror = rand(rng, Bool),
            offset = rand(rng, 0:95),
        ) for _ in 1:episode_count
    ]
end

function generated_seed_pairs(plan, count::Integer)
    count >= 0 || throw(ArgumentError("count must be non-negative."))
    rng = StableRNG(Int(plan["master_seed"]))
    existing_count = Int(plan["generated_pairs"])
    for _ in 1:existing_count
        rand(rng, 1:2_000_000_000)
        rand(rng, 1:2_000_000_000)
    end
    return [
        (run_seed = rand(rng, 1:2_000_000_000), ic_seed = rand(rng, 1:2_000_000_000))
        for _ in 1:count
    ]
end

function make_entry(seeds; origin::Symbol, batch_id, trace, import_metadata = nothing)
    return (
        run_id = run_id(seeds.run_seed, seeds.ic_seed),
        run_seed = Int(seeds.run_seed),
        ic_seed = Int(seeds.ic_seed),
        origin = origin,
        batch_id = string(batch_id),
        created_at = string(now()),
        varying_trace = normalize_trace(trace),
        import_metadata = import_metadata,
    )
end

entry_batch(entry) = hasproperty(entry, :batch_id) ? entry.batch_id : "legacy_$(entry.run_id)"

function entry_batches(entries)
    order = String[]
    grouped = Dict{String, Vector{Any}}()
    for entry in entries
        id = entry_batch(entry)
        haskey(grouped, id) || push!(order, id)
        push!(get!(grouped, id, Any[]), entry)
    end
    return [grouped[id] for id in order]
end

collect_plan_entries(results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY) =
    collect(load_plan(results_directory)["entries"])

function read_result_summary(path::AbstractString)
    isfile(path) || return nothing
    return try
        JLD2.jldopen(path, "r") do file
            (
                status = string(read(file, "status")),
                protocol = Symbol(read(file, "protocol")),
                algorithm = haskey(file, "algorithm") ? Symbol(read(file, "algorithm")) : :mat,
                run_seed = Int(read(file, "run_seed")),
                ic_seed = Int(read(file, "ic_seed")),
                episode_target = Int(read(file, "episode_target")),
                episodes_completed = Int(read(file, "episodes_completed")),
            )
        end
    catch
        nothing
    end
end

function complete_result(path, entry, protocol, algorithm)
    metadata = read_result_summary(path)
    isnothing(metadata) && return false
    expected = DEFAULT_EPISODES[protocol]
    metadata.status == "complete" || return false
    metadata.protocol == protocol || error("Protocol mismatch in '$path'.")
    metadata.algorithm == algorithm || error("Algorithm mismatch in '$path'.")
    metadata.run_seed == entry.run_seed || error("Run-seed mismatch in '$path'.")
    metadata.ic_seed == entry.ic_seed || error("IC-seed mismatch in '$path'.")
    metadata.episode_target == expected || error("Episode-target mismatch in '$path'.")
    metadata.episodes_completed == expected || error("Incomplete result marked complete: '$path'.")
    return true
end

function complete_validation(path, entry, protocol, algorithm)
    isfile(path) || return false
    return try
        JLD2.jldopen(path, "r") do file
            string(read(file, "status")) == "complete" &&
            string(read(file, "run_id")) == entry.run_id &&
            Symbol(read(file, "protocol")) == protocol &&
            Symbol(read(file, "algorithm")) == algorithm &&
            Int(read(file, "run_seed")) == entry.run_seed &&
            Int(read(file, "ic_seed")) == entry.ic_seed
        end
    catch
        false
    end
end

function is_pending(entry, protocols, results_directory)
    return any(protocols) do protocol
        any(ALGORITHMS) do algorithm
            target = result_path(results_directory, entry.run_id, protocol, algorithm)
            !complete_result(
                target,
                entry,
                protocol,
                algorithm,
            ) || !complete_validation(
                validation_path(results_directory, entry.run_id, protocol, algorithm),
                entry,
                protocol,
                algorithm,
            )
        end
    end
end

function p3_metadata(path::AbstractString)
    return JLD2.jldopen(path, "r") do file
        trace = haskey(file, "initial_condition_trace") ?
            normalize_trace(read(file, "initial_condition_trace")) : NamedTuple[]
        rewards = read(file, "rewards")
        (
            path = path,
            status = string(read(file, "status")),
            protocol = Symbol(read(file, "protocol")),
            config_name = Symbol(read(file, "config_name")),
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
            episode_target = Int(read(file, "episode_target")),
            episodes_completed = Int(read(file, "episodes_completed")),
            reward_count = length(rewards),
            trace = trace,
        )
    end
end

function find_package3_pairs(package3_directory::AbstractString)
    by_seed = Dict{Tuple{Int, Int}, Dict{Symbol, Any}}()
    isdir(package3_directory) || return by_seed
    for protocol in PROTOCOLS
        root = joinpath(package3_directory, string(protocol))
        isdir(root) || continue
        for (directory, _, files) in walkdir(root)
            "modified_full.jld2" in files || continue
            path = joinpath(directory, "modified_full.jld2")
            metadata = try
                p3_metadata(path)
            catch error_value
                @warn "Ignoring unreadable Package-3 result" path exception = error_value
                continue
            end
            metadata.protocol == protocol || error("P3 protocol mismatch in '$path'.")
            metadata.status == "complete" || continue
            metadata.config_name == :modified_full || continue
            expected = DEFAULT_EPISODES[protocol]
            metadata.episode_target == expected || continue
            metadata.episodes_completed == expected || continue
            metadata.reward_count == expected || continue
            protocol === :varying && length(metadata.trace) != expected && continue
            records = get!(
                by_seed,
                (metadata.run_seed, metadata.ic_seed),
                Dict{Symbol, Any}(),
            )
            if haskey(records, protocol)
                source_hash(records[protocol].path) == source_hash(metadata.path) || error(
                    "Conflicting P3 results share seeds $(metadata.run_seed)/$(metadata.ic_seed) " *
                    "for protocol $protocol.",
                )
            else
                records[protocol] = metadata
            end
        end
    end
    return filter(pair -> all(haskey(last(pair), protocol) for protocol in PROTOCOLS), by_seed)
end

function copy_import_result(source, target, entry, protocol)
    source_sha = source_hash(source)
    if isfile(target)
        metadata = read_result_summary(target)
        if !isnothing(metadata) && metadata.status == "complete" &&
           metadata.run_seed == entry.run_seed && metadata.ic_seed == entry.ic_seed
            compatible = JLD2.jldopen(target, "r") do file
                haskey(file, "package4_source_sha256") &&
                read(file, "package4_source_sha256") == source_sha &&
                read(file, "package4_run_id") == entry.run_id &&
                Symbol(read(file, "package4_protocol")) == protocol
            end
            compatible && return target
        end
        error("Refusing to overwrite incompatible result '$target'.")
    end

    mkpath(dirname(target))
    temporary = target * ".tmp.$(getpid())"
    rm(temporary; force = true)
    cp(source, temporary; force = false)
    JLD2.jldopen(temporary, "a+") do file
        file["algorithm"] = :mat
        file["package4_schema_version"] = SCHEMA_VERSION
        file["package4_run_id"] = entry.run_id
        file["package4_batch_id"] = entry_batch(entry)
        file["package4_origin"] = :imported_package3
        file["package4_imported_at"] = string(now())
        file["package4_source_path"] = abspath(source)
        file["package4_source_sha256"] = source_sha
        file["package4_protocol"] = protocol
    end
    mv(temporary, target; force = false)
    return target
end

function import_package3!(;
    count::Integer,
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    package3_directory::AbstractString = DEFAULT_PACKAGE3_DIRECTORY,
    preview::Bool = false,
)
    count >= 0 || throw(ArgumentError("count must be non-negative."))
    return with_plan_lock(results_directory) do
        plan = load_plan(results_directory; allow_missing = true)
        entries = plan["entries"]
        existing_seeds = Set((entry.run_seed, entry.ic_seed) for entry in entries)
        existing_imports = [entry for entry in entries if entry.origin === :imported_package3]
        pending_batches = [
            batch for batch in entry_batches(existing_imports)
            if any(entry -> is_pending(entry, collect(PROTOCOLS), results_directory), batch)
        ]
        if !isempty(pending_batches)
            selected = first(pending_batches)
            length(selected) == count || error(
                "The unfinished import batch contains $(length(selected)) seed pairs, but " *
                "--n-runs requested $count. Resume it with the original count.",
            )
            return selected
        end
        needed = count
        pairs = find_package3_pairs(package3_directory)
        candidates = [
            (seeds = seeds, records = records) for (seeds, records) in sort!(collect(pairs); by = first)
            if !(seeds in existing_seeds)
        ]
        length(candidates) >= needed || error(
            "Requested $count imported seed pairs, but only $(length(candidates)) new complete " *
            "fixed/varying P3 pairs are available. " *
            "No replacement seeds were generated.",
        )

        selected = Any[]
        batch_id = "imported_$(length(existing_imports) + 1)_$(Dates.format(now(), dateformat"yyyymmddTHHMMSS"))"
        for candidate in candidates[1:needed]
            run_seed, ic_seed = candidate.seeds
            varying = candidate.records[:varying]
            fixed = candidate.records[:fixed]
            fixed.run_seed == varying.run_seed == run_seed || error("P3 run-seed pair mismatch.")
            fixed.ic_seed == varying.ic_seed == ic_seed || error("P3 IC-seed pair mismatch.")
            metadata = Dict(
                "fixed_source" => abspath(fixed.path),
                "varying_source" => abspath(varying.path),
                "fixed_sha256" => source_hash(fixed.path),
                "varying_sha256" => source_hash(varying.path),
            )
            entry = make_entry(
                (; run_seed, ic_seed);
                origin = :imported_package3,
                batch_id,
                trace = varying.trace,
                import_metadata = metadata,
            )
            push!(selected, entry)
            preview && continue
            for protocol in PROTOCOLS
                source = candidate.records[protocol].path
                target = result_path(results_directory, entry.run_id, protocol, :mat)
                copy_import_result(source, target, entry, protocol)
            end
            push!(entries, entry)
        end
        preview || save_plan(results_directory, plan)
        return selected
    end
end

function ensure_generated_entries!(plan, count, protocols, results_directory; preview = false)
    entries = plan["entries"]
    generated = [entry for entry in entries if entry.origin === :generated]
    pending_batches = [
        batch for batch in entry_batches(generated)
        if any(entry -> is_pending(entry, protocols, results_directory), batch)
    ]
    if !isempty(pending_batches)
        selected = first(pending_batches)
        length(selected) == count || error(
            "The unfinished generated batch contains $(length(selected)) seed pairs, but " *
            "--n-runs requested $count. Resume it with the original count.",
        )
        return selected
    end

    selected = Any[]
    needed = count
    occupied = Set((entry.run_seed, entry.ic_seed) for entry in entries)
    batch_id = "generated_$(Int(plan["generated_pairs"]) + 1)_$(Dates.format(now(), dateformat"yyyymmddTHHMMSS"))"
    for seeds in generated_seed_pairs(plan, needed)
        (seeds.run_seed, seeds.ic_seed) in occupied && error(
            "Generated seed-pair collision; the plan was not changed.",
        )
        entry = make_entry(
            seeds;
            origin = :generated,
            batch_id,
            trace = varying_schedule(seeds.ic_seed, DEFAULT_EPISODES[:varying]),
        )
        push!(selected, entry)
        push!(occupied, (entry.run_seed, entry.ic_seed))
        preview || push!(entries, entry)
    end
    preview || (plan["generated_pairs"] = Int(plan["generated_pairs"]) + needed)
    return selected
end

function jobs_for(entries, protocols, results_directory; force::Bool = false)
    jobs = NamedTuple[]
    for entry in entries, protocol in protocols, algorithm in ALGORITHMS
        target = result_path(results_directory, entry.run_id, protocol, algorithm)
        isdir(target * ".lock") && continue
        if force || !complete_result(target, entry, protocol, algorithm)
            push!(jobs, (
                task = :train,
                run_id = entry.run_id,
                protocol = protocol,
                algorithm = algorithm,
                path = target,
            ))
        elseif !complete_validation(
            validation_path(results_directory, entry.run_id, protocol, algorithm),
            entry,
            protocol,
            algorithm,
        )
            isdir(validation_path(results_directory, entry.run_id, protocol, algorithm) * ".lock") &&
                continue
            # A newly imported MAT checkpoint is validated by the paired IPPO
            # training worker, so the normal import launch still has only one
            # worker per protocol and seed pair.
            paired_ippo = result_path(results_directory, entry.run_id, protocol, :ippo)
            if algorithm === :mat && entry.origin === :imported_package3 &&
               !complete_result(paired_ippo, entry, protocol, :ippo)
                continue
            end
            push!(jobs, (
                task = :validate,
                run_id = entry.run_id,
                protocol = protocol,
                algorithm = algorithm,
                path = target,
            ))
        end
    end
    return jobs
end

function prepare_jobs(;
    n_runs::Integer,
    look_for_imports::Bool,
    protocol = :all,
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    package3_directory::AbstractString = DEFAULT_PACKAGE3_DIRECTORY,
    preview::Bool = false,
    overwrite::Bool = false,
)
    n_runs >= 0 || throw(ArgumentError("n_runs must be non-negative."))
    protocols = selected_protocols(protocol)
    if overwrite
        plan = load_plan(results_directory)
        origin = look_for_imports ? :imported_package3 : :generated
        candidates = [entry for entry in plan["entries"] if entry.origin === origin]
        length(candidates) >= n_runs || error(
            "Cannot overwrite $n_runs $origin entries; only $(length(candidates)) exist.",
        )
        entries = n_runs == 0 ? Any[] : candidates[(end - n_runs + 1):end]
        return (; entries, jobs = jobs_for(entries, protocols, results_directory; force = true))
    end
    if look_for_imports
        entries = import_package3!(;
            count = n_runs,
            results_directory,
            package3_directory,
            preview,
        )
        return (; entries, jobs = jobs_for(entries, protocols, results_directory))
    end

    return with_plan_lock(results_directory) do
        plan = load_plan(results_directory; allow_missing = true)
        entries = ensure_generated_entries!(
            plan,
            n_runs,
            protocols,
            results_directory;
            preview,
        )
        preview || save_plan(results_directory, plan)
        return (; entries, jobs = jobs_for(entries, protocols, results_directory))
    end
end

function run_file_path(protocol::Symbol, algorithm::Symbol)
    prefix = protocol === :fixed ? "FixedIC" : "VaryingIC"
    suffix = algorithm === :mat ? "MAT" : "IPPO"
    return joinpath(REVISION_DIRECTORY, "Run_Files", "$(prefix)_$(suffix).jl")
end

function include_run_file!(protocol, algorithm, run_seed, bootstrap_directory)
    ENV["REVISION_RUN_SEED"] = string(run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = bootstrap_directory
    mkpath(bootstrap_directory)
    Base.include(@__MODULE__, run_file_path(protocol, algorithm))
    return
end

function configure_agent!(protocol, algorithm, entry)
    global agent = nothing
    global hook = nothing
    global env = nothing
    global rng = StableRNG(entry.run_seed)
    if protocol === :varying
        global initial_condition_rng = StableRNG(entry.ic_seed)
        global initial_condition_split = :train
    end
    if algorithm === :mat
        global useSeparateValueChain = MAT_CONFIG.useSeparateValueChain
        global useLayerNorm = MAT_CONFIG.useLayerNorm
        global useSelfAttentionFirst = MAT_CONFIG.useSelfAttentionFirst
        global use_mus = MAT_CONFIG.use_mus
    end
    Random.seed!(entry.run_seed)
    Base.invokelatest(initialize_setup)
    Random.seed!(entry.run_seed)
    hook.is_display_on_exit = false
    hook.display_after_episode = false
    return
end

function configure_training_initializers!(protocol, entry, observed_trace)
    if protocol === :fixed
        hook.generate_random_init = () -> Base.invokelatest(generate_random_init)
        return
    end
    index = Ref(0)
    hook.generate_random_init = () -> begin
        index[] += 1
        index[] <= length(entry.varying_trace) || error("IC schedule exhausted.")
        choice = entry.varying_trace[index[]]
        result, split, base_seed, mirror, offset = Base.invokelatest(
            generate_random_init;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        )
        used = (; split, base_seed, mirror, offset)
        used == choice || error("Run file did not reproduce the planned initial condition.")
        push!(observed_trace, used)
        return result
    end
    return
end

function train_exact_episodes!(episode_count::Integer; progress_every::Integer = 0)
    episode_count >= 0 || throw(ArgumentError("episode_count must be non-negative."))
    episode_count == 0 && return
    hook(PRE_EXPERIMENT_STAGE, agent, env)
    agent(PRE_EXPERIMENT_STAGE, env)
    for episode in 1:episode_count
        reset!(env)
        agent(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, agent, env)
        while !(is_terminated(env) || is_truncated(env))
            action = agent(env)
            agent(PRE_ACT_STAGE, env, action)
            hook(PRE_ACT_STAGE, agent, env, action)
            env(action)
            agent(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, agent, env)
        end
        agent(POST_EPISODE_STAGE, env)
        hook(POST_EPISODE_STAGE, agent, env)
        if progress_every > 0 && (episode % progress_every == 0 || episode == episode_count)
            @printf("[%s] episode %d/%d, reward %.6g\n", now(), episode, episode_count, hook.rewards[end])
            flush(stdout)
        end
    end
    hook(POST_EXPERIMENT_STAGE, agent, env)
    length(hook.rewards) == episode_count || error("Episode count mismatch.")
    return
end

function parameter_count(algorithm)
    models = algorithm === :mat ?
        (agent.policy.encoder, agent.policy.decoder) :
        (agent.policy.approximator.actor, agent.policy.approximator.critic)
    return sum(length, Flux.trainables(models))
end

function parameter_hash(algorithm)
    models = algorithm === :mat ?
        (agent.policy.encoder, agent.policy.decoder) :
        (agent.policy.approximator.actor, agent.policy.approximator.critic)
    io = IOBuffer()
    for parameter in Flux.trainables(models)
        array = Array(parameter)
        write(io, string(eltype(array)), UInt8(0), string(size(array)), UInt8(0))
        write(io, reinterpret(UInt8, vec(array)))
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function run_parameter_snapshot()
    result = Dict{String, Any}()
    for name in SNAPSHOT_GLOBALS
        isdefined(@__MODULE__, name) || continue
        result[string(name)] = getfield(@__MODULE__, name)
    end
    return result
end

function source_hashes(protocol, algorithm)
    hashes = Dict(
        "run_file" => source_hash(run_file_path(protocol, algorithm)),
        "experiment_runner" => source_hash(@__FILE__),
    )
    agent_source = algorithm === :mat ? "agent_mat.jl" : "agent_ppo.jl"
    hashes[agent_source] = source_hash(joinpath(RL_ROOT, "src", agent_source))
    return hashes
end

function save_training_result(path, entry, protocol, algorithm, episode_target,
                              elapsed_seconds, started_at, observed_trace, initial_parameter_hash)
    count = parameter_count(algorithm)
    expected = EXPECTED_PARAMETERS[(protocol, algorithm)]
    count == expected || error("Parameter count $count does not match expected $expected.")
    atomic_jldsave(
        path;
        schema_version = SCHEMA_VERSION,
        status = "complete",
        run_id = entry.run_id,
        protocol = protocol,
        algorithm = algorithm,
        config_name = algorithm === :mat ? :modified_full : :parameter_sharing_ippo,
        config = algorithm === :mat ? MAT_CONFIG : nothing,
        origin = entry.origin,
        batch_id = entry_batch(entry),
        run_seed = entry.run_seed,
        ic_seed = entry.ic_seed,
        episode_target = Int(episode_target),
        episodes_completed = length(hook.rewards),
        elapsed_seconds = Float64(elapsed_seconds),
        started_at = string(started_at),
        completed_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        run_file = run_file_path(protocol, algorithm),
        source_hashes = source_hashes(protocol, algorithm),
        run_parameters = run_parameter_snapshot(),
        parameter_count = count,
        initial_parameter_hash = initial_parameter_hash,
        final_parameter_hash = parameter_hash(algorithm),
        planned_initial_condition_trace = protocol === :varying ? entry.varying_trace : NamedTuple[],
        initial_condition_trace = observed_trace,
        rewards = copy(hook.rewards),
        rewards_all_timesteps = copy(hook.rewards_all_timesteps),
        rewards_compare = copy(hook.rewards_compare),
        errored_episodes = copy(hook.errored_episodes),
        best_reward = hook.bestreward,
        best_episode = hook.bestepisode,
        agent = agent,
    )
end

function save_failure(path, entry, protocol, algorithm, episode_target, started_at, error_message)
    active_hook = isdefined(@__MODULE__, :hook) ? hook : nothing
    active_agent = isdefined(@__MODULE__, :agent) ? agent : nothing
    rewards = !isnothing(active_hook) && hasproperty(active_hook, :rewards) ?
        copy(active_hook.rewards) : Float64[]
    atomic_jldsave(
        path;
        schema_version = SCHEMA_VERSION,
        status = "failed",
        run_id = entry.run_id,
        batch_id = entry_batch(entry),
        protocol = protocol,
        algorithm = algorithm,
        origin = entry.origin,
        run_seed = entry.run_seed,
        ic_seed = entry.ic_seed,
        episode_target = Int(episode_target),
        episodes_completed = length(rewards),
        started_at = string(started_at),
        failed_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        error_message = error_message,
        rewards = rewards,
        agent = active_agent,
    )
end

function set_initial_condition!(protocol, choice = nothing)
    state_value = if protocol === :fixed
        Base.invokelatest(generate_random_init)
    else
        result = Base.invokelatest(
            generate_random_init;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        )
        first(result)
    end
    env.y0 = state_value
    reset!(env)
    return
end

function deterministic_rollout(protocol, choice = nothing)
    set_initial_condition!(protocol, choice)
    total = 0.0
    while !(is_terminated(env) || is_truncated(env))
        distributions = RL.prob(agent.policy, env)
        action = copy(getproperty(distributions, :μ))
        hasproperty(agent.policy, :clip1) && agent.policy.clip1 && clamp!(action, -1.0, 1.0)
        env(action)
        total += mean(reward(env))
    end
    return total
end

function validation_cases(protocol)
    protocol === :fixed && return [nothing]
    raw = JLD2.load(CORPUS_PATH, "corpus")
    split = haskey(raw, :validation) ? raw[:validation] : raw["validation"]
    base_seeds = sort!(Int.(collect(keys(split))))
    length(base_seeds) == 1 || error("Expected one validation base seed, got $(length(base_seeds)).")
    return [
        (split = :validation, base_seed = only(base_seeds), mirror = mirror, offset = offset)
        for mirror in (false, true), offset in (0, 20)
    ] |> vec
end

function validate_agent!(path, entry, protocol, algorithm, training_path)
    Flux.testmode!(agent.policy)
    cases = validation_cases(protocol)
    scores = [deterministic_rollout(protocol, choice) for choice in cases]
    atomic_jldsave(
        path;
        schema_version = SCHEMA_VERSION,
        status = "complete",
        run_id = entry.run_id,
        batch_id = entry_batch(entry),
        protocol = protocol,
        algorithm = algorithm,
        run_seed = entry.run_seed,
        ic_seed = entry.ic_seed,
        policy = "deterministic_mean_action",
        training_result_path = abspath(training_path),
        training_result_sha256 = source_hash(training_path),
        validation_cases = cases,
        validation_scores = scores,
        validation_mean = mean(scores),
        completed_at = string(now()),
    )
    return path
end

function save_validation_failure(path, entry, protocol, algorithm, training_path, error_message)
    atomic_jldsave(
        path;
        schema_version = SCHEMA_VERSION,
        status = "failed",
        run_id = entry.run_id,
        protocol = protocol,
        algorithm = algorithm,
        run_seed = entry.run_seed,
        ic_seed = entry.ic_seed,
        training_result_path = abspath(training_path),
        failed_at = string(now()),
        error_message = error_message,
    )
end

function find_entry(results_directory, id)
    matches = [entry for entry in load_plan(results_directory)["entries"] if entry.run_id == id]
    length(matches) == 1 || error("Expected one plan entry for '$id', found $(length(matches)).")
    return only(matches)
end

function run_worker(;
    task = :train,
    run_id,
    protocol,
    algorithm,
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    episodes::Union{Nothing, Integer} = nothing,
    overwrite::Bool = false,
)
    task = Symbol(lowercase(string(task)))
    task in (:train, :validate) || throw(ArgumentError("task must be train or validate."))
    protocol = normalize_protocol(protocol)
    algorithm = normalize_algorithm(algorithm)
    entry = find_entry(results_directory, string(run_id))
    episode_target = something(episodes, DEFAULT_EPISODES[protocol])
    training_path = result_path(results_directory, entry.run_id, protocol, algorithm)
    validation_target = validation_path(results_directory, entry.run_id, protocol, algorithm)
    lock_target = task === :train ? training_path * ".lock" : validation_target * ".lock"
    lock = acquire_lock(lock_target)
    try
        include_run_file!(
            protocol,
            algorithm,
            entry.run_seed,
            joinpath(dirname(training_path), "_runfile_$(algorithm)"),
        )
        configure_agent!(protocol, algorithm, entry)

        if task === :train
            if !overwrite && complete_result(training_path, entry, protocol, algorithm)
                @printf("Skipping complete training result %s\n", training_path)
                global agent = JLD2.load(training_path, "agent")
            else
                started_at = now()
                observed_trace = NamedTuple[]
                try
                    count = parameter_count(algorithm)
                    expected = EXPECTED_PARAMETERS[(protocol, algorithm)]
                    count == expected || error(
                        "Parameter count $count does not match expected $expected before training.",
                    )
                    initial_parameter_hash = parameter_hash(algorithm)
                    configure_training_initializers!(protocol, entry, observed_trace)
                    progress_every = episode_target == 0 ? 0 : max(1, episode_target ÷ 20)
                    elapsed = @elapsed Base.invokelatest(
                        train_exact_episodes!,
                        episode_target;
                        progress_every,
                    )
                    protocol === :varying && observed_trace != entry.varying_trace[1:episode_target] &&
                        error("Observed IC trace differs from the plan.")
                    save_training_result(
                        training_path,
                        entry,
                        protocol,
                        algorithm,
                        episode_target,
                        elapsed,
                        started_at,
                        observed_trace,
                        initial_parameter_hash,
                    )
                catch error_value
                    message = sprint(showerror, error_value, catch_backtrace())
                    save_failure(
                        training_path,
                        entry,
                        protocol,
                        algorithm,
                        episode_target,
                        started_at,
                        message,
                    )
                    rethrow(error_value)
                end
            end
        else
            complete_result(training_path, entry, protocol, algorithm) || error(
                "Cannot validate missing or incomplete result '$training_path'.",
            )
            global agent = JLD2.load(training_path, "agent")
        end

        if overwrite || !complete_validation(validation_target, entry, protocol, algorithm)
            try
                Base.invokelatest(
                    validate_agent!,
                    validation_target,
                    entry,
                    protocol,
                    algorithm,
                    training_path,
                )
            catch error_value
                save_validation_failure(
                    validation_target,
                    entry,
                    protocol,
                    algorithm,
                    training_path,
                    sprint(showerror, error_value, catch_backtrace()),
                )
                rethrow(error_value)
            end
        end

        if algorithm === :ippo && entry.origin === :imported_package3
            mat_training = result_path(results_directory, entry.run_id, protocol, :mat)
            mat_validation = validation_path(results_directory, entry.run_id, protocol, :mat)
            if complete_result(mat_training, entry, protocol, :mat) &&
               !complete_validation(mat_validation, entry, protocol, :mat)
                global agent = JLD2.load(mat_training, "agent")
                try
                    Base.invokelatest(
                        validate_agent!,
                        mat_validation,
                        entry,
                        protocol,
                        :mat,
                        mat_training,
                    )
                catch error_value
                    save_validation_failure(
                        mat_validation,
                        entry,
                        protocol,
                        :mat,
                        mat_training,
                        sprint(showerror, error_value, catch_backtrace()),
                    )
                    rethrow(error_value)
                end
            end
        end
        return training_path
    finally
        rm(lock; recursive = true, force = true)
    end
end

end
