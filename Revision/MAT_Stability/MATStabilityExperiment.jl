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

export MAT_CONFIGS, run_worker, seed_plan

const SCHEMA_VERSION = 1
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

function initial_condition_probe(protocol::Symbol)
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

    result, split, base_seed, mirror, offset = Base.invokelatest(
        generate_random_init;
        split = :train,
        rng = deepcopy(initial_condition_rng),
    )
    return (; state_hash = array_hash(result), split, base_seed, mirror, offset)
end

function configure_episode_initializer!(protocol::Symbol, ic_trace)
    if protocol === :fixed
        hook.generate_random_init = () -> Base.invokelatest(generate_random_init)
        return
    end

    hook.generate_random_init = () -> begin
        result, split, base_seed, mirror, offset = Base.invokelatest(
            generate_random_init;
            split = :train,
            rng = initial_condition_rng,
        )
        push!(ic_trace, (; split, base_seed, mirror, offset))
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
        (
            status = read(file, "status"),
            protocol = Symbol(read(file, "protocol")),
            replicate = Int(read(file, "replicate")),
            config_name = Symbol(read(file, "config_name")),
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
            episode_target = Int(read(file, "episode_target")),
            episodes_completed = Int(read(file, "episodes_completed")),
            shared_initial_hash = read(file, "shared_initial_hash"),
            full_initial_hash = read(file, "full_initial_hash"),
            policy_rng_probe = read(file, "policy_rng_probe"),
            initial_condition_probe = read(file, "initial_condition_probe"),
            initial_condition_trace = read(file, "initial_condition_trace"),
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
    return true
end

function check_pairing!(
    metadata,
    config,
    protocol,
    shared_hash_reference,
    rng_probe_reference,
    ic_probe_reference,
    ic_trace_reference,
    modified_half_full_hash,
)
    if isnothing(shared_hash_reference)
        shared_hash_reference = metadata.shared_initial_hash
        rng_probe_reference = metadata.policy_rng_probe
        ic_probe_reference = metadata.initial_condition_probe
    else
        metadata.shared_initial_hash == shared_hash_reference || error(
            "Shared initial MAT parameters differ for config $(config.name).",
        )
        metadata.policy_rng_probe == rng_probe_reference || error(
            "Policy RNG state differs for config $(config.name).",
        )
        metadata.initial_condition_probe == ic_probe_reference || error(
            "Initial-condition probe differs for config $(config.name).",
        )
    end

    if protocol === :varying
        if isnothing(ic_trace_reference)
            ic_trace_reference = metadata.initial_condition_trace
        else
            metadata.initial_condition_trace == ic_trace_reference || error(
                "Varying-IC trace differs for config $(config.name).",
            )
        end
    end

    if config.name === :modified_half
        modified_half_full_hash = metadata.full_initial_hash
    elseif config.name === :modified_full
        metadata.full_initial_hash == modified_half_full_hash || error(
            "modified_half and modified_full do not have identical initial networks.",
        )
    end

    return (
        shared_hash_reference,
        rng_probe_reference,
        ic_probe_reference,
        ic_trace_reference,
        modified_half_full_hash,
    )
end

function acquire_worker_lock(directory::AbstractString)
    lock_directory = joinpath(directory, ".worker.lock")
    try
        mkdir(lock_directory)
    catch error_value
        isdir(lock_directory) || rethrow(error_value)
        error(
            "Worker lock already exists at '$lock_directory'. " *
            "Another worker may be active. Remove only stale locks after checking tmux.",
        )
    end
    open(joinpath(lock_directory, "owner.txt"), "w") do io
        println(io, "pid=$(getpid())")
        println(io, "host=$(gethostname())")
        println(io, "started_at=$(now())")
    end
    return lock_directory
end

function run_worker(
    protocol_input,
    replicate::Integer;
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
    episode_target = dry_run ? 0 : something(episodes, DEFAULT_EPISODES[protocol])
    episode_target >= 0 || throw(ArgumentError("episodes must be non-negative."))
    seeds = seed_plan(replicate)

    result_root = dry_run ? joinpath(results_directory, "dry_run") : results_directory
    replicate_directory = joinpath(
        result_root,
        string(protocol),
        @sprintf("replicate_%02d", replicate),
    )
    mkpath(replicate_directory)
    lock_directory = acquire_worker_lock(replicate_directory)
    failures = Pair{Symbol, String}[]

    try
        include_run_file!(
            protocol,
            seeds.run_seed,
            joinpath(replicate_directory, "_runfile"),
        )

        shared_hash_reference = nothing
        rng_probe_reference = nothing
        ic_probe_reference = nothing
        ic_trace_reference = nothing
        modified_half_full_hash = nothing

        for config in MAT_CONFIGS
            result_path = joinpath(replicate_directory, string(config.name) * ".jld2")
            metadata = nothing

            if isfile(result_path) && !overwrite
                existing = read_result_metadata(result_path)
                if validate_existing_result(
                    existing,
                    protocol,
                    replicate,
                    config,
                    seeds,
                    episode_target,
                )
                    @printf("Skipping complete result %s\n", result_path)
                    metadata = existing
                end
            end

            if isnothing(metadata)
                started_at = now()
                ic_trace = NamedTuple[]
                try
                    @printf(
                        "Starting %s/%02d/%s with run_seed=%d, ic_seed=%d, episodes=%d\n",
                        protocol,
                        replicate,
                        config.name,
                        seeds.run_seed,
                        seeds.ic_seed,
                        episode_target,
                    )
                    flush(stdout)

                    configure_agent!(config, protocol, seeds)
                    hashes = initial_hashes()
                    hashes.value_chain_matches_main || error(
                        "Separate value chain is not initialized from the main chain.",
                    )
                    hashes.value_chain_is_independent || error(
                        "Separate value chain shares parameter arrays with the main chain.",
                    )

                    policy_rng_probe = rand(deepcopy(agent.policy.rng), UInt64, 4)
                    ic_probe = initial_condition_probe(protocol)
                    configure_episode_initializer!(protocol, ic_trace)

                    progress_every = episode_target == 0 ? 0 : max(1, episode_target ÷ 20)
                    elapsed_seconds = @elapsed Base.invokelatest(
                        train_exact_episodes!,
                        episode_target;
                        progress_every,
                    )

                    atomic_save_complete(
                        result_path;
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
                    metadata = read_result_metadata(result_path)
                    @printf("Completed and saved %s\n", result_path)
                    flush(stdout)
                catch error_value
                    backtrace = catch_backtrace()
                    error_message = sprint(showerror, error_value, backtrace)
                    atomic_save_failure(
                        result_path;
                        protocol,
                        replicate,
                        config,
                        seeds,
                        episode_target,
                        started_at,
                        error_message,
                    )
                    push!(failures, config.name => error_message)
                    @error "MAT stability config failed" protocol replicate config.name exception = (
                        error_value,
                        backtrace,
                    )
                    continue
                end
            end

            (
                shared_hash_reference,
                rng_probe_reference,
                ic_probe_reference,
                ic_trace_reference,
                modified_half_full_hash,
            ) = check_pairing!(
                metadata,
                config,
                protocol,
                shared_hash_reference,
                rng_probe_reference,
                ic_probe_reference,
                ic_trace_reference,
                modified_half_full_hash,
            )
        end
    finally
        rm(lock_directory; recursive = true, force = true)
    end

    isempty(failures) || error(
        "Worker finished with failed configs: " *
        join(["$(name): $(message)" for (name, message) in failures], "\n"),
    )
    return replicate_directory
end

end
