module MATStabilityExperiment

using Dates
using Flux
using JLD2
using Printf
using Random
using RL
using SHA
using Sockets
using StableRNGs

export MAT_CONFIGS, DEFAULT_EPISODES, DEFAULT_RESULTS_DIRECTORY, prepare_jobs,
       run_worker, seed_plan, result_path, plan_path

const SCHEMA_VERSION = 1
const PLAN_SCHEMA_VERSION = 1
const SEED_PLAN_MASTER = 20260730
const REPLICATES = 1:5
const DEFAULT_EPISODES = Dict(:fixed => 2_000, :varying => 4_000)

const MAT_CONFIGS = (
    (
        name = :python_like,
        useSeparateValueChain = false,
        useLayerNorm = true,
        useSelfAttentionFirst = true,
        use_mus = false,
    ),
    (
        name = :modified_half,
        useSeparateValueChain = true,
        useLayerNorm = false,
        useSelfAttentionFirst = false,
        use_mus = false,
    ),
    (
        name = :modified_full,
        useSeparateValueChain = true,
        useLayerNorm = false,
        useSelfAttentionFirst = false,
        use_mus = true,
    ),
)

const REVISION_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = normpath(joinpath(REVISION_DIRECTORY, ".."))
const RL_ROOT = normpath(joinpath(dirname(pathof(RL)), ".."))
const DEFAULT_RESULTS_DIRECTORY = joinpath(@__DIR__, "results")
const CORPUS_PATH = joinpath(
    REVISION_DIRECTORY,
    "VaryingIC_Corpus",
    "varying_ic_corpus.jld2",
)
const LOADED_PROTOCOL = Ref{Union{Nothing, Symbol}}(nothing)
const SNAPSHOT_GLOBALS = (
    :Ra,
    :Pr,
    :Nx,
    :Nz,
    :Lx,
    :Lz,
    :inner_dt,
    :dt,
    :actuators,
    :sensors,
    :update_freq,
    :learning_rate,
    :n_epochs,
    :n_microbatches,
    :clip_range,
    :target_kl,
    :dim_model,
    :block_num,
    :head_num,
    :head_dim,
    :ffn_dim,
    :drop_out,
    :betas,
    :positional_encoding,
    :positional_encoding_decoder,
    :customCrossAttention,
    :jointPPO,
    :one_by_one_training,
    :useSeparateValueChain,
    :useLayerNorm,
    :useSelfAttentionFirst,
    :use_mus,
    :randomIC,
    :joon_pe,
    :square_rewards,
)

# The selected run file is included at runtime. Predeclaring its public bindings
# avoids Julia 1.12 world-age issues when the already compiled worker methods
# access objects and methods populated by that include.
agent = nothing
hook = nothing
env = nothing
model = nothing
function initialize_setup end
function generate_random_init end

function normalize_protocol(protocol)::Symbol
    normalized = Symbol(lowercase(string(protocol)))
    normalized in keys(DEFAULT_EPISODES) || throw(
        ArgumentError("Unknown protocol '$protocol'. Use :fixed or :varying."),
    )
    return normalized
end

function selected_protocols(value)
    selection = Symbol(lowercase(string(value)))
    selection === :all && return [:fixed, :varying]
    return [normalize_protocol(selection)]
end

function config_for(name)
    normalized = Symbol(lowercase(string(name)))
    matches = [config for config in MAT_CONFIGS if config.name === normalized]
    length(matches) == 1 || throw(
        ArgumentError(
            "Unknown MAT configuration '$name'. Use " *
            join(string.(getproperty.(MAT_CONFIGS, :name)), ", ") * ".",
        ),
    )
    return only(matches)
end

function seed_plan(replicate::Integer)
    replicate in REPLICATES || throw(
        ArgumentError("Replicate must be in $(first(REPLICATES)):$(last(REPLICATES))."),
    )

    planner = StableRNG(SEED_PLAN_MASTER)
    run_seed = 0
    ic_seed = 0
    for _ in 1:replicate
        run_seed = rand(planner, 1:2_000_000_000)
        ic_seed = rand(planner, 1:2_000_000_000)
    end
    return (; run_seed, ic_seed)
end

plan_path(results_directory::AbstractString) = joinpath(results_directory, "run_plan.jld2")

result_path(results_directory::AbstractString, protocol, replicate::Integer, config) = joinpath(
    results_directory,
    string(normalize_protocol(protocol)),
    @sprintf("replicate_%02d", replicate),
    string(config_for(config).name) * ".jld2",
)

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

function varying_schedule(
    ic_seed::Integer,
    episode_count::Integer,
    seeds = training_seeds(),
)
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

function expected_plan()
    training_seed_values = training_seeds()
    entries = [
        begin
            seeds = seed_plan(replicate)
            (
                replicate = Int(replicate),
                run_seed = seeds.run_seed,
                ic_seed = seeds.ic_seed,
                varying_trace = varying_schedule(
                    seeds.ic_seed,
                    DEFAULT_EPISODES[:varying],
                    training_seed_values,
                ),
            )
        end for replicate in REPLICATES
    ]
    return Dict{String, Any}(
        "schema_version" => PLAN_SCHEMA_VERSION,
        "seed_plan_master" => SEED_PLAN_MASTER,
        "replicates" => collect(REPLICATES),
        "configs" => collect(getproperty.(MAT_CONFIGS, :name)),
        "default_episodes" => copy(DEFAULT_EPISODES),
        "entries" => entries,
        "created_at" => string(now()),
    )
end

function validate_plan(plan; verify_varying_trace::Bool = false)
    Int(plan["schema_version"]) == PLAN_SCHEMA_VERSION || error(
        "Unsupported MAT-stability plan schema.",
    )
    Int(plan["seed_plan_master"]) == SEED_PLAN_MASTER || error(
        "MAT-stability plan uses a different seed-plan master.",
    )
    Symbol.(collect(plan["configs"])) == collect(getproperty.(MAT_CONFIGS, :name)) || error(
        "MAT-stability plan has different configurations.",
    )
    Int.(collect(plan["replicates"])) == collect(REPLICATES) || error(
        "MAT-stability plan has different replicates.",
    )
    stored_episodes = plan["default_episodes"]
    all(
        Int(stored_episodes[protocol]) == episode_count
        for (protocol, episode_count) in DEFAULT_EPISODES
    ) || error("MAT-stability plan has different episode budgets.")
    stored_entries = collect(plan["entries"])
    length(stored_entries) == length(REPLICATES) || error(
        "MAT-stability plan has the wrong number of replicate entries.",
    )
    training_seed_values = verify_varying_trace ? training_seeds() : Int[]
    for (replicate, stored) in zip(REPLICATES, stored_entries)
        seeds = seed_plan(replicate)
        Int(stored.replicate) == replicate || error("Plan replicate order differs.")
        Int(stored.run_seed) == seeds.run_seed || error("Plan run seed differs.")
        Int(stored.ic_seed) == seeds.ic_seed || error("Plan IC seed differs.")
        stored_trace = normalize_trace(stored.varying_trace)
        length(stored_trace) == DEFAULT_EPISODES[:varying] || error(
            "Plan Varying-IC trace has the wrong length for replicate $replicate.",
        )
        if verify_varying_trace
            stored_trace == varying_schedule(
                seeds.ic_seed,
                DEFAULT_EPISODES[:varying],
                training_seed_values,
            ) || error(
                "Plan Varying-IC trace differs for replicate $replicate.",
            )
        end
    end
    return plan
end

function atomic_jldsave(path::AbstractString; values...)
    mkpath(dirname(path))
    temporary_path = path * ".tmp.$(getpid()).$(time_ns())"
    try
        JLD2.jldsave(temporary_path; values...)
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return path
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

function load_or_create_plan(
    results_directory::AbstractString;
    preview::Bool = false,
    verify_varying_trace::Bool = false,
)
    path = plan_path(results_directory)
    if isfile(path)
        return validate_plan(
            JLD2.load(path, "plan");
            verify_varying_trace,
        )
    end
    plan = expected_plan()
    preview && return plan
    lock = acquire_lock(path * ".lock")
    try
        if isfile(path)
            return validate_plan(
                JLD2.load(path, "plan");
                verify_varying_trace,
            )
        end
        atomic_jldsave(path; plan)
        return validate_plan(plan; verify_varying_trace = true)
    finally
        rm(lock; recursive = true, force = true)
    end
end

function find_plan_entry(results_directory::AbstractString, replicate::Integer)
    path = plan_path(results_directory)
    isfile(path) || error(
        "Frozen MAT-stability plan '$path' is missing. Run prepare_runs.jl or " *
        "launch_tmux.sh before starting workers.",
    )
    plan = validate_plan(JLD2.load(path, "plan"))
    matches = [entry for entry in plan["entries"] if Int(entry.replicate) == replicate]
    length(matches) == 1 || error("Expected one plan entry for replicate $replicate.")
    entry = only(matches)
    return (
        replicate = Int(entry.replicate),
        run_seed = Int(entry.run_seed),
        ic_seed = Int(entry.ic_seed),
        varying_trace = normalize_trace(entry.varying_trace),
    )
end

function run_file_path(protocol::Symbol)
    filename = protocol === :fixed ? "FixedIC_MAT.jl" : "VaryingIC_MAT.jl"
    return joinpath(REVISION_DIRECTORY, "Run_Files", filename)
end

function include_run_file!(protocol::Symbol, run_seed::Int, bootstrap_directory::AbstractString)
    if !isnothing(LOADED_PROTOCOL[])
        LOADED_PROTOCOL[] === protocol || error(
            "This process already loaded protocol $(LOADED_PROTOCOL[]). " *
            "Use one process per protocol and replicate.",
        )
        return
    end

    ENV["REVISION_RUN_SEED"] = string(run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = bootstrap_directory
    mkpath(bootstrap_directory)
    Base.include(@__MODULE__, run_file_path(protocol))
    LOADED_PROTOCOL[] = protocol
    return
end

function write_array!(io::IO, values)
    array = Array(values)
    write(io, string(eltype(array)))
    write(io, UInt8(0))
    write(io, string(size(array)))
    write(io, UInt8(0))
    write(io, reinterpret(UInt8, vec(array)))
    return
end

function parameter_hash(models...)
    io = IOBuffer()
    parameters = Flux.trainables(models)
    write(io, string(length(parameters)))
    write(io, UInt8(0))
    for parameter in parameters
        write_array!(io, parameter)
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

source_hash(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function source_hashes(protocol::Symbol)
    return Dict(
        "run_file" => source_hash(run_file_path(protocol)),
        "agent_mat" => source_hash(joinpath(RL_ROOT, "src", "agent_mat.jl")),
        "experiment_runner" => source_hash(@__FILE__),
    )
end

function run_parameter_snapshot()
    snapshot = Dict{String, Any}()
    for name in SNAPSHOT_GLOBALS
        isdefined(@__MODULE__, name) || continue
        snapshot[string(name)] = getfield(@__MODULE__, name)
    end
    return snapshot
end

function array_hash(values)
    io = IOBuffer()
    write_array!(io, values)
    return bytes2hex(SHA.sha256(take!(io)))
end

function initial_hashes()
    encoder = agent.policy.encoder
    decoder = agent.policy.decoder
    shared_hash = parameter_hash(
        encoder.embedding,
        encoder.position_encoding,
        encoder.ln,
        encoder.blocks,
        encoder.head,
        decoder,
    )
    full_hash = parameter_hash(encoder, decoder)
    shared_parameter_count = sum(
        length,
        Flux.trainables((
            encoder.embedding,
            encoder.position_encoding,
            encoder.ln,
            encoder.blocks,
            encoder.head,
            decoder,
        )),
    )
    full_parameter_count = sum(length, Flux.trainables((encoder, decoder)))

    value_chain_is_independent = true
    value_chain_matches_main = true
    if encoder.useSeparateValueChain
        main_parameters = Flux.trainables((
            encoder.embedding,
            encoder.position_encoding,
            encoder.ln,
            encoder.blocks,
        ))
        value_parameters = Flux.trainables((
            encoder.embedding_v,
            encoder.position_encoding_v,
            encoder.ln_v,
            encoder.blocks_v,
        ))
        value_chain_matches_main =
            length(main_parameters) == length(value_parameters) &&
            all(main == value for (main, value) in zip(main_parameters, value_parameters))
        value_chain_is_independent =
            length(main_parameters) == length(value_parameters) &&
            all(main !== value for (main, value) in zip(main_parameters, value_parameters))
    end

    return (;
        shared_hash,
        full_hash,
        shared_parameter_count,
        full_parameter_count,
        value_chain_matches_main,
        value_chain_is_independent,
    )
end

function configure_agent!(config, protocol::Symbol, seeds)
    global agent = nothing
    global hook = nothing
    global env = nothing
    global useSeparateValueChain = config.useSeparateValueChain
    global useLayerNorm = config.useLayerNorm
    global useSelfAttentionFirst = config.useSelfAttentionFirst
    global use_mus = config.use_mus
    global rng = StableRNG(seeds.run_seed)

    if protocol === :varying
        global initial_condition_rng = StableRNG(seeds.ic_seed)
        global initial_condition_split = :train
    end

    Random.seed!(seeds.run_seed)
    Base.invokelatest(initialize_setup)
    Random.seed!(seeds.run_seed)

    hook.is_display_on_exit = false
    hook.display_after_episode = false
    return
end

function initial_condition_probe(protocol::Symbol, planned_trace)
    if protocol === :fixed
        result = Base.invokelatest(generate_random_init)
        return (
            state_hash = array_hash(result),
            split = :fixed,
            base_seed = nothing,
            mirror = false,
            offset = 0,
        )
    end

    isempty(planned_trace) && error("Varying-IC plan has no episodes.")
    choice = first(planned_trace)
    result, split, base_seed, mirror, offset = Base.invokelatest(
        generate_random_init;
        split = choice.split,
        base_seed = choice.base_seed,
        mirror = choice.mirror,
        offset = choice.offset,
    )
    observed = (; split, base_seed, mirror, offset)
    observed == choice || error("Run file did not reproduce the planned IC probe.")
    return merge((; state_hash = array_hash(result)), observed)
end

function configure_episode_initializer!(protocol::Symbol, ic_trace, planned_trace)
    if protocol === :fixed
        hook.generate_random_init = () -> Base.invokelatest(generate_random_init)
        return
    end

    hook.generate_random_init = () -> begin
        index = length(ic_trace) + 1
        index <= length(planned_trace) || error(
            "Varying-IC plan exhausted after $(length(planned_trace)) episodes.",
        )
        choice = planned_trace[index]
        result, split, base_seed, mirror, offset = Base.invokelatest(
            generate_random_init;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        )
        observed = (; split, base_seed, mirror, offset)
        observed == choice || error("Run file did not reproduce planned Varying IC $choice.")
        push!(ic_trace, observed)
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
            @printf(
                "[%s] episode %d/%d, latest reward %.6g\n",
                Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"),
                episode,
                episode_count,
                hook.rewards[end],
            )
            flush(stdout)
        end
    end

    hook(POST_EXPERIMENT_STAGE, agent, env)
    length(hook.rewards) == episode_count || error(
        "Expected $episode_count completed episodes, got $(length(hook.rewards)).",
    )
    return
end

function repository_state(path::AbstractString)
    isdir(joinpath(path, ".git")) || return (commit = "unknown", dirty = "unknown")
    commit = try
        readchomp(`git -C $path rev-parse HEAD`)
    catch
        "unknown"
    end
    dirty = try
        isempty(readchomp(`git -C $path status --porcelain`)) ? "clean" : "dirty"
    catch
        "unknown"
    end
    return (; commit, dirty)
end

function atomic_save_complete(
    path::AbstractString;
    protocol,
    replicate,
    config,
    seeds,
    episode_target,
    elapsed_seconds,
    started_at,
    hashes,
    policy_rng_probe,
    ic_probe,
    ic_trace,
)
    mkpath(dirname(path))
    temporary_path = path * ".tmp.$(getpid())"
    rm(temporary_path; force = true)

    project_state = repository_state(PROJECT_ROOT)
    rl_state = repository_state(RL_ROOT)
    JLD2.jldsave(
        temporary_path;
        schema_version = SCHEMA_VERSION,
        status = "complete",
        protocol = protocol,
        replicate = Int(replicate),
        config_name = config.name,
        config = config,
        run_seed = seeds.run_seed,
        ic_seed = seeds.ic_seed,
        seed_plan_master = SEED_PLAN_MASTER,
        episode_target = Int(episode_target),
        episodes_completed = length(hook.rewards),
        elapsed_seconds = Float64(elapsed_seconds),
        started_at = string(started_at),
        completed_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        run_file = run_file_path(protocol),
        source_hashes = source_hashes(protocol),
        run_parameters = run_parameter_snapshot(),
        project_git_commit = project_state.commit,
        project_git_state = project_state.dirty,
        rl_git_commit = rl_state.commit,
        rl_git_state = rl_state.dirty,
        shared_initial_hash = hashes.shared_hash,
        full_initial_hash = hashes.full_hash,
        shared_parameter_count = hashes.shared_parameter_count,
        full_parameter_count = hashes.full_parameter_count,
        value_chain_matches_main = hashes.value_chain_matches_main,
        value_chain_is_independent = hashes.value_chain_is_independent,
        policy_rng_probe = policy_rng_probe,
        initial_condition_probe = ic_probe,
        initial_condition_trace = ic_trace,
        rewards = copy(hook.rewards),
        rewards_all_timesteps = copy(hook.rewards_all_timesteps),
        rewards_compare = copy(hook.rewards_compare),
        errored_episodes = copy(hook.errored_episodes),
        best_reward = hook.bestreward,
        best_episode = hook.bestepisode,
        agent = agent,
    )
    mv(temporary_path, path; force = true)
    return path
end

function atomic_save_failure(
    path::AbstractString;
    protocol,
    replicate,
    config,
    seeds,
    episode_target,
    started_at,
    error_message,
)
    mkpath(dirname(path))
    temporary_path = path * ".tmp.$(getpid())"
    rm(temporary_path; force = true)
    active_hook = isdefined(@__MODULE__, :hook) ? hook : nothing
    partial_rewards =
        !isnothing(active_hook) && hasproperty(active_hook, :rewards) ?
        copy(active_hook.rewards) : Float64[]
    partial_timestep_rewards =
        !isnothing(active_hook) && hasproperty(active_hook, :rewards_all_timesteps) ?
        copy(active_hook.rewards_all_timesteps) : Float64[]

    JLD2.jldsave(
        temporary_path;
        schema_version = SCHEMA_VERSION,
        status = "failed",
        protocol = protocol,
        replicate = Int(replicate),
        config_name = config.name,
        config = config,
        run_seed = seeds.run_seed,
        ic_seed = seeds.ic_seed,
        seed_plan_master = SEED_PLAN_MASTER,
        episode_target = Int(episode_target),
        episodes_completed = length(partial_rewards),
        started_at = string(started_at),
        failed_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        run_file = run_file_path(protocol),
        source_hashes = source_hashes(protocol),
        run_parameters = run_parameter_snapshot(),
        error_message = error_message,
        rewards = partial_rewards,
        rewards_all_timesteps = partial_timestep_rewards,
    )
    mv(temporary_path, path; force = true)
    return path
end

function read_result_metadata(path::AbstractString)
    return JLD2.jldopen(path, "r") do file
        status = string(read(file, "status"))
        (
            status,
            protocol = Symbol(read(file, "protocol")),
            replicate = Int(read(file, "replicate")),
            config_name = Symbol(read(file, "config_name")),
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
            episode_target = Int(read(file, "episode_target")),
            episodes_completed = Int(read(file, "episodes_completed")),
            shared_initial_hash = status == "complete" ?
                read(file, "shared_initial_hash") : nothing,
            full_initial_hash = status == "complete" ? read(file, "full_initial_hash") : nothing,
            policy_rng_probe = status == "complete" ? read(file, "policy_rng_probe") : nothing,
            initial_condition_probe = status == "complete" ?
                read(file, "initial_condition_probe") : nothing,
            initial_condition_trace = status == "complete" ?
                read(file, "initial_condition_trace") : NamedTuple[],
            source_hashes = status == "complete" && haskey(file, "source_hashes") ?
                read(file, "source_hashes") : nothing,
        )
    end
end

function validate_existing_result(metadata, protocol, replicate, config, seeds, episode_target)
    metadata.status == "complete" || return false
    metadata.protocol == protocol || error("Existing result has the wrong protocol.")
    metadata.replicate == replicate || error("Existing result has the wrong replicate.")
    metadata.config_name == config.name || error("Existing result has the wrong config.")
    metadata.run_seed == seeds.run_seed || error("Existing result has the wrong run seed.")
    metadata.ic_seed == seeds.ic_seed || error("Existing result has the wrong IC seed.")
    metadata.episode_target == episode_target || error(
        "Existing result has episode target $(metadata.episode_target), expected $episode_target.",
    )
    metadata.episodes_completed == episode_target || error(
        "Existing result is marked complete but contains " *
        "$(metadata.episodes_completed)/$episode_target episodes.",
    )
    metadata.source_hashes == source_hashes(protocol) || return false
    return true
end

function prepare_jobs(;
    protocol = :all,
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    dry_run::Bool = false,
    overwrite::Bool = false,
    preview::Bool = false,
)
    plan = load_or_create_plan(
        results_directory;
        preview,
        verify_varying_trace = true,
    )
    result_root = dry_run ? joinpath(results_directory, "dry_run") : results_directory
    jobs = NamedTuple[]
    for selected_protocol in selected_protocols(protocol)
        episode_target = dry_run ? 0 : DEFAULT_EPISODES[selected_protocol]
        for entry in plan["entries"]
            replicate = Int(entry.replicate)
            seeds = (
                run_seed = Int(entry.run_seed),
                ic_seed = Int(entry.ic_seed),
            )
            for config in MAT_CONFIGS
                path = result_path(result_root, selected_protocol, replicate, config.name)
                complete = false
                if isfile(path) && !overwrite
                    metadata = read_result_metadata(path)
                    complete = validate_existing_result(
                        metadata,
                        selected_protocol,
                        replicate,
                        config,
                        seeds,
                        episode_target,
                    )
                end
                complete && continue
                push!(jobs, (
                    protocol = selected_protocol,
                    replicate,
                    config_name = config.name,
                    run_seed = seeds.run_seed,
                    ic_seed = seeds.ic_seed,
                    episode_target,
                    path = abspath(path),
                ))
            end
        end
    end
    return (; plan, jobs)
end

function run_worker(
    protocol_input,
    replicate::Integer;
    config,
    episodes::Union{Nothing, Integer} = nothing,
    dry_run::Bool = false,
    overwrite::Bool = false,
    results_directory::AbstractString = get(
        ENV,
        "MAT_STABILITY_RESULTS_DIR",
        DEFAULT_RESULTS_DIRECTORY,
    ),
)
    protocol = normalize_protocol(protocol_input)
    replicate in REPLICATES || throw(
        ArgumentError("Replicate must be in $(first(REPLICATES)):$(last(REPLICATES))."),
    )
    selected_config = config_for(config)
    episode_target = dry_run ? 0 : something(episodes, DEFAULT_EPISODES[protocol])
    episode_target >= 0 || throw(ArgumentError("episodes must be non-negative."))
    entry = find_plan_entry(results_directory, replicate)
    seeds = (run_seed = entry.run_seed, ic_seed = entry.ic_seed)
    planned_trace = if protocol === :varying
        episode_target <= length(entry.varying_trace) || error(
            "Requested $episode_target Varying episodes, but the frozen plan contains " *
            "only $(length(entry.varying_trace)).",
        )
        entry.varying_trace[1:episode_target]
    else
        NamedTuple[]
    end

    result_root = dry_run ? joinpath(results_directory, "dry_run") : results_directory
    result_target = result_path(result_root, protocol, replicate, selected_config.name)
    replicate_directory = dirname(result_target)
    mkpath(replicate_directory)
    lock_directory = acquire_lock(result_target * ".lock")

    try
        if isfile(result_target) && !overwrite
            existing = read_result_metadata(result_target)
            if validate_existing_result(
                existing,
                protocol,
                replicate,
                selected_config,
                seeds,
                episode_target,
            )
                @printf("Skipping complete result %s\n", result_target)
                return result_target
            end
        end

        include_run_file!(
            protocol,
            seeds.run_seed,
            joinpath(replicate_directory, "_runfile_$(selected_config.name)"),
        )

        started_at = now()
        ic_trace = NamedTuple[]
        try
            @printf(
                "Starting %s/%02d/%s with run_seed=%d, ic_seed=%d, episodes=%d\n",
                protocol,
                replicate,
                selected_config.name,
                seeds.run_seed,
                seeds.ic_seed,
                episode_target,
            )
            flush(stdout)

            configure_agent!(selected_config, protocol, seeds)
            hashes = initial_hashes()
            hashes.value_chain_matches_main || error(
                "Separate value chain is not initialized from the main chain.",
            )
            hashes.value_chain_is_independent || error(
                "Separate value chain shares parameter arrays with the main chain.",
            )

            policy_rng_probe = rand(deepcopy(agent.policy.rng), UInt64, 4)
            ic_probe = initial_condition_probe(protocol, entry.varying_trace)
            configure_episode_initializer!(protocol, ic_trace, planned_trace)

            progress_every = episode_target == 0 ? 0 : max(1, episode_target ÷ 20)
            elapsed_seconds = @elapsed Base.invokelatest(
                train_exact_episodes!,
                episode_target;
                progress_every,
            )
            protocol === :varying && ic_trace != planned_trace && error(
                "Observed Varying-IC trace differs from the frozen plan.",
            )

            atomic_save_complete(
                result_target;
                protocol,
                replicate,
                config = selected_config,
                seeds,
                episode_target,
                elapsed_seconds,
                started_at,
                hashes,
                policy_rng_probe,
                ic_probe,
                ic_trace,
            )
            @printf("Completed and saved %s\n", result_target)
            flush(stdout)
        catch error_value
            backtrace = catch_backtrace()
            error_message = sprint(showerror, error_value, backtrace)
            atomic_save_failure(
                result_target;
                protocol,
                replicate,
                config = selected_config,
                seeds,
                episode_target,
                started_at,
                error_message,
            )
            @error "MAT stability config failed" protocol replicate selected_config.name exception = (
                error_value,
                backtrace,
            )
            rethrow(error_value)
        end
    finally
        rm(lock_directory; recursive = true, force = true)
    end

    return result_target
end

end
