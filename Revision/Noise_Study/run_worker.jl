using Dates
using JLD2
using Random
using SHA
using StableRNGs
using Statistics

include(joinpath(@__DIR__, "NoiseStudy.jl"))
using .NoiseStudy

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DISTILLATION_ROOT = joinpath(PROJECT_ROOT, "Revision", "Expert_Apprentice_Distillation")

function usage()
    println("""
    Usage: julia --startup-file=no --project=. Revision/Noise_Study/run_worker.jl [options]

      --manifest PATH
      --protocol fixed|varying
      --controller expert|sparse|c_match
      --noise-level 0|0.01|0.05|0.10|0.20|0.30|0.40|0.50
      --results-dir PATH
      --retry-failed
      --help

    One worker owns one protocol/controller/noise-level combination. Clean
    workers import the frozen baseline artifacts once. Noisy workers run all
    ten replicates and all protocol test cases sequentially and resume from
    atomic per-episode files.
    """)
end

function parse_options(arguments)
    values = Dict{String, Any}(
        "manifest" => nothing,
        "protocol" => nothing,
        "controller" => nothing,
        "noise_level" => nothing,
        "results_dir" => joinpath(@__DIR__, "results"),
        "retry_failed" => false,
    )
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            usage()
            return nothing
        elseif argument == "--retry-failed"
            values["retry_failed"] = true
            index += 1
            continue
        end
        startswith(argument, "--") || error("Unknown argument '$argument'.")
        index < length(arguments) || error("Missing value after $argument.")
        key = replace(argument[3:end], "-" => "_")
        haskey(values, key) || error("Unknown option '$argument'.")
        values[key] = arguments[index + 1]
        index += 2
    end
    for required in ("manifest", "protocol", "controller", "noise_level")
        isnothing(values[required]) && error("--$(replace(required, "_" => "-")) is required.")
    end
    return (;
        manifest_path = abspath(string(values["manifest"])),
        protocol = normalize_protocol(values["protocol"]),
        controller = normalize_controller(values["controller"]),
        noise_level = normalize_noise_level(parse(Float64, string(values["noise_level"]))),
        results_root = abspath(string(values["results_dir"])),
        retry_failed = Bool(values["retry_failed"]),
    )
end

function valueof(value, key::Symbol)
    value isa NamedTuple && return getproperty(value, key)
    value isa AbstractDict && return haskey(value, key) ? value[key] : value[string(key)]
    return getproperty(value, key)
end

latest_runtime_binding(name::Symbol) = Base.invokelatest(() -> getfield(@__MODULE__, name))

function controller_from_manifest(manifest, controller::Symbol)
    matches = filter(manifest[:controllers]) do raw
        Symbol(valueof(raw, :kind)) === controller
    end
    length(matches) == 1 || error("Expected one $controller controller in the manifest.")
    return symbolize(only(matches))
end

function validate_manifest(options, manifest)
    Symbol(manifest[:protocol]) === options.protocol || error("Worker/manifest protocol mismatch.")
    if !(options.noise_level in Float64.(manifest[:noise_levels]))
        @warn "Noise level is absent from the existing manifest; continuing with a supported level added after manifest creation." noise_level=options.noise_level manifest_path=options.manifest_path
    end
    Int(manifest[:replicate_count]) == NOISE_REPLICATES || error("Unexpected replicate count in manifest.")
    Int(manifest[:test_steps]) == NOISE_TEST_STEPS || error("Unexpected test horizon in manifest.")
    return controller_from_manifest(manifest, options.controller)
end

function output_directory(options, manifest)
    worker_output_directory(
        options.results_root,
        string(manifest[:experiment_id]),
        options.protocol,
        options.controller,
        options.noise_level,
    )
end

function status_is_complete(path, options, manifest_fingerprint)
    isfile(path) || return false
    try
        status = JLD2.load(path)
        return Symbol(status["state"]) === :complete &&
               Symbol(status["protocol"]) === options.protocol &&
               Symbol(status["controller"]) === options.controller &&
               Float64(status["noise_level"]) == options.noise_level &&
               string(status["manifest_fingerprint"]) == string(manifest_fingerprint)
    catch
        return false
    end
end

function write_worker_status!(path, options, manifest; state::Symbol, message = "", entries...)
    atomic_save(
        path;
        schema_version = NOISE_SCHEMA_VERSION,
        experiment = :package10_sensor_noise,
        experiment_id = string(manifest[:experiment_id]),
        protocol = options.protocol,
        controller = options.controller,
        noise_level = options.noise_level,
        manifest_fingerprint = string(manifest[:manifest_fingerprint]),
        state,
        message,
        updated_at = string(Dates.now(Dates.UTC)),
        entries...,
    )
end

function case_choice(case)
    choice = valueof(case, :choice)
    isnothing(choice) && return nothing
    return (
        split = Symbol(valueof(choice, :split)),
        base_seed = Int(valueof(choice, :base_seed)),
        mirror = Bool(valueof(choice, :mirror)),
        offset = Int(valueof(choice, :offset)),
    )
end

case_identity(choice) = isnothing(choice) ? (:fixed_shared,) :
    (choice.split, choice.base_seed, choice.mirror, choice.offset)

function normalize_episode(case, rewards, state_nusselt, actions; source_path)
    normalized_rewards = Float64.(rewards)
    normalized_nusselt = Float64.(state_nusselt)
    normalized_actions = Float32.(actions)
    length(normalized_rewards) == NOISE_TEST_STEPS || error("Clean reward length mismatch in '$source_path'.")
    length(normalized_nusselt) == NOISE_TEST_STEPS || error("Clean state_Nu length mismatch in '$source_path'.")
    size(normalized_actions) == (NOISE_TEST_STEPS, 12) || error("Clean action shape mismatch in '$source_path'.")
    all(isfinite, normalized_rewards) && all(isfinite, normalized_nusselt) && all(isfinite, normalized_actions) ||
        error("Clean baseline contains non-finite values: $source_path")
    return (
        case_id = string(valueof(case, :case_id)),
        choice = case_choice(case),
        rewards = normalized_rewards,
        state_nusselt = normalized_nusselt,
        actions = normalized_actions,
        simulation_times = Float64[],
        source_path = abspath(source_path),
    )
end

function verify_source_file(path, expected_hash)
    isfile(path) || error("Clean source is missing: $path")
    observed = file_sha256(path)
    string(observed) == string(expected_hash) || error("Clean source hash mismatch: $path")
    return abspath(path)
end

function load_expert_clean_episodes(controller, cases)
    source = symbolize(controller[:clean_source])
    path = verify_source_file(string(source[:path]), source[:sha256])
    loaded = JLD2.load(path)
    raw_episodes = loaded["episodes"]
    by_identity = Dict(case_identity(valueof(episode, :choice)) => episode for episode in raw_episodes)
    return [begin
        choice = case_choice(case)
        raw = by_identity[case_identity(choice)]
        normalize_episode(case, raw.rewards, raw.state_nusselt, raw.actions; source_path = path)
    end for case in cases]
end

function load_sparse_clean_episodes(controller, cases, protocol::Symbol)
    source = symbolize(controller[:clean_source])
    path = verify_source_file(string(source[:path]), source[:sha256])
    loaded = JLD2.load(path)
    if protocol === :fixed
        only_case = only(cases)
        return [normalize_episode(
            only_case,
            loaded["rewards"],
            loaded["state_nusselt"],
            loaded["actions"];
            source_path = path,
        )]
    end
    by_identity = Dict(
        case_identity((split = Symbol(raw.split), base_seed = Int(raw.base_seed), mirror = Bool(raw.mirror), offset = Int(raw.offset))) => raw
        for raw in loaded["episodes"]
    )
    return [begin
        raw = by_identity[case_identity(case_choice(case))]
        normalize_episode(case, raw.rewards, raw.state_nusselt, raw.actions; source_path = path)
    end for case in cases]
end

function load_c_match_clean_episodes(controller, cases)
    source = symbolize(controller[:clean_source])
    paths = String.(source[:paths])
    hashes = String.(source[:sha256])
    length(paths) == length(hashes) || error("C_match clean source hash count mismatch.")
    by_identity = Dict{Any, Any}()
    for (path, expected_hash) in zip(paths, hashes)
        verified = verify_source_file(path, expected_hash)
        loaded = JLD2.load(verified)
        string(loaded["controller_id"]) == string(controller[:controller_id]) || error(
            "C_match clean cache controller mismatch: $verified",
        )
        choice = loaded["case_spec"]
        normalized_choice = isnothing(choice) ? nothing : (
            split = Symbol(choice.split),
            base_seed = Int(choice.base_seed),
            mirror = Bool(choice.mirror),
            offset = Int(choice.offset),
        )
        by_identity[case_identity(normalized_choice)] = (
            loaded = loaded,
            path = verified,
        )
    end
    return [begin
        raw = by_identity[case_identity(case_choice(case))]
        normalize_episode(
            case,
            raw.loaded["rewards"],
            raw.loaded["global_nusselt"],
            raw.loaded["actions"];
            source_path = raw.path,
        )
    end for case in cases]
end

function load_clean_episodes(controller, cases, protocol::Symbol)
    kind = Symbol(controller[:kind])
    kind === :expert && return load_expert_clean_episodes(controller, cases)
    kind === :sparse && return load_sparse_clean_episodes(controller, cases, protocol)
    kind === :c_match && return load_c_match_clean_episodes(controller, cases)
    error("Unsupported clean controller $kind.")
end

function episode_is_current(path, options, manifest, replicate, case)
    isfile(path) || return false
    try
        loaded = JLD2.load(path)
        return Int(loaded["schema_version"]) == NOISE_SCHEMA_VERSION &&
               Symbol(loaded["protocol"]) === options.protocol &&
               Symbol(loaded["controller"]) === options.controller &&
               Float64(loaded["noise_level"]) == options.noise_level &&
               Int(loaded["replicate"]) == replicate &&
               string(loaded["case_id"]) == string(valueof(case, :case_id)) &&
               string(loaded["manifest_fingerprint"]) == string(manifest[:manifest_fingerprint]) &&
               length(loaded["rewards"]) == NOISE_TEST_STEPS &&
               length(loaded["state_nusselt"]) == NOISE_TEST_STEPS
    catch
        return false
    end
end

function save_episode(path, options, manifest, controller, case, episode; replicate, seed)
    rewards = Float64.(episode.rewards)
    nusselt = Float64.(episode.state_nusselt)
    actions = Float32.(episode.actions)
    atomic_save(
        path;
        schema_version = NOISE_SCHEMA_VERSION,
        experiment = :package10_sensor_noise_episode,
        experiment_id = string(manifest[:experiment_id]),
        protocol = options.protocol,
        controller = options.controller,
        controller_id = string(controller[:controller_id]),
        configuration = string(controller[:configuration]),
        noise_model = :additive_iid_zero_mean_gaussian,
        noise_level = options.noise_level,
        channel_scales = Float64.(valueof(manifest[:channel_scales], :scales)),
        positional_encoding_noiseless = true,
        replicate,
        noise_seed = seed,
        case_index = Int(valueof(case, :index)),
        case_id = string(valueof(case, :case_id)),
        choice = case_choice(case),
        steps = NOISE_TEST_STEPS,
        rewards,
        state_nusselt = nusselt,
        actions,
        simulation_times = Float64.(episode.simulation_times),
        reward_sum = sum(rewards),
        mean_reward = mean(rewards),
        sum_state_nusselt = sum(nusselt),
        mean_state_nusselt = mean(nusselt),
        action_saturation_fraction = count(value -> abs(value) >= 1.0f0 - 1.0f-6, actions) / length(actions),
        clean_source_path = hasproperty(episode, :source_path) ? episode.source_path : "",
        manifest_path = options.manifest_path,
        manifest_sha256 = file_sha256(options.manifest_path),
        manifest_fingerprint = string(manifest[:manifest_fingerprint]),
        completed_at = string(Dates.now(Dates.UTC)),
    )
    return abspath(path)
end

function import_clean_baseline!(options, manifest, controller, output)
    cases = manifest[:cases]
    episodes = load_clean_episodes(controller, cases, options.protocol)
    length(episodes) == length(cases) || error("Clean source did not provide every test case.")
    paths = String[]
    for (case, episode) in zip(cases, episodes)
        path = episode_output_path(output, 0.0, 0, string(valueof(case, :case_id)))
        if !episode_is_current(path, options, manifest, 0, case)
            println("Importing clean baseline: $(options.controller) / $(valueof(case, :case_id))")
            save_episode(path, options, manifest, controller, case, episode; replicate = 0, seed = nothing)
        end
        push!(paths, abspath(path))
    end
    return paths
end

function configure_runtime!(options, manifest, controller, output)
    expert = controller_from_manifest(manifest, :expert)
    ENV["DISTILLATION_PROTOCOL"] = string(options.protocol)
    ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
    ENV["DISTILLATION_GROUP_CHANNELS"] = "false"
    ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "false"
    ENV[options.protocol === :fixed ? "DISTILLATION_FIXED_EXPERT_PATH" : "DISTILLATION_VARYING_EXPERT_PATH"] =
        string(expert[:checkpoint_path])
    ENV["REVISION_RUN_SEED"] = string(NOISE_MASTER_SEED)
    ENV["REVISION_RUN_DIRECTORY"] = joinpath(output, "runtime")
    ENV["DISTILLATION_OUTPUT_DIRECTORY"] = joinpath(output, "apprentice_output")
    Base.include(@__MODULE__, joinpath(DISTILLATION_ROOT, "Expert_Apprentice.jl"))
    Bool(latest_runtime_binding(:joon_pe)) || error("Noise Study requires the frozen Joon positional encoding.")
    expert_metadata = latest_runtime_binding(:DISTILLATION_EXPERT_METADATA)
    string(expert_metadata[:identifier]) == string(manifest[:expert_identifier]) || error(
        "Runtime expert identity differs from the frozen manifest.",
    )
    model = if options.controller === :expert
        latest_runtime_binding(:agent).policy
    else
        path = string(controller[:checkpoint_path])
        file_sha256(path) == string(controller[:checkpoint_sha256]) || error("Controller checkpoint hash mismatch.")
        JLD2.load(path)["model_payload"]
    end
    runtime_flux = latest_runtime_binding(:Flux)
    Base.invokelatest(runtime_flux.testmode!, model)
    mask = options.controller === :expert ? nothing : Float32.(controller[:input_mask])
    isnothing(mask) || length(mask) == size(latest_runtime_binding(:env).state, 1) || error("Controller mask length mismatch.")
    return (; model, mask)
end

function initialize_case!(protocol::Symbol, case)
    initialize = latest_runtime_binding(:generate_random_init)
    choice = case_choice(case)
    if protocol === :fixed
        Base.invokelatest(initialize)
    else
        Base.invokelatest(
            initialize;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        )
    end
    runtime_rl = latest_runtime_binding(:RL)
    runtime_env = latest_runtime_binding(:env)
    Base.invokelatest(runtime_rl.reset!, runtime_env)
    return nothing
end

function noisy_local_observation(noise_rng, noise_level, channel_scales)
    runtime_env = latest_runtime_binding(:env)
    sensor_indices = latest_runtime_binding(:sensor_positions)
    actuator_indices = latest_runtime_binding(:actuators_to_sensors)
    window = latest_runtime_binding(:window_size)
    global_observation_function = latest_runtime_binding(:global_sensor_observation)
    local_observation_function = latest_runtime_binding(:local_mat_observation)
    physical = Base.invokelatest(
        global_observation_function,
        runtime_env.y;
        horizontal_indices = sensor_indices[1],
        vertical_indices = sensor_indices[2],
        add_joon_position_encoding = false,
    )
    encoded = noisy_encoded_global_observation(physical, noise_rng, noise_level, channel_scales)
    return Base.invokelatest(
        local_observation_function,
        encoded;
        actuator_sensor_indices = actuator_indices,
        window_size = window,
    )
end

function normalize_action(action)
    values = Float32.(Array(action))
    ndims(values) == 3 && size(values, 3) == 1 && (values = dropdims(values; dims = 3))
    length(values) == 12 || error("Expected twelve actions, got $(size(values)).")
    return vec(values)
end

function controller_action(runtime, observation)
    runtime_rl = latest_runtime_binding(:RL)
    input = isnothing(runtime.mask) ? observation : observation .* runtime.mask
    action = Base.invokelatest(runtime_rl.prob, runtime.model, input, nothing).μ
    return ndims(action) == 3 ? action[:, :, 1] : action
end

function run_noisy_episode(options, manifest, runtime, case, seed; steps::Int = NOISE_TEST_STEPS)
    steps >= 1 || throw(ArgumentError("steps must be positive."))
    initialize_case!(options.protocol, case)
    runtime_env = latest_runtime_binding(:env)
    nusselt_function = latest_runtime_binding(:state_Nu)
    noise_rng = StableRNG(seed)
    scales = Float64.(valueof(manifest[:channel_scales], :scales))
    rewards = Vector{Float64}(undef, steps)
    nusselt = Vector{Float64}(undef, steps)
    simulation_times = Vector{Float64}(undef, steps)
    actions = Matrix{Float32}(undef, steps, 12)
    for step in 1:steps
        observation = Base.invokelatest(noisy_local_observation, noise_rng, options.noise_level, scales)
        action = Base.invokelatest(controller_action, runtime, observation)
        actions[step, :] .= normalize_action(action)
        Base.invokelatest(runtime_env, action)
        rewards[step] = mean(Float64.(runtime_env.reward))
        nusselt[step] = Float64(Base.invokelatest(nusselt_function, runtime_env))
        simulation = latest_runtime_binding(:simulation)
        simulation_times[step] = Float64(simulation.model.clock.time)
        isfinite(rewards[step]) && isfinite(nusselt[step]) && all(isfinite, @view actions[step, :]) || error(
            "Non-finite value in $(valueof(case, :case_id)), replicate $seed, step $step.",
        )
    end
    return (; rewards, state_nusselt = nusselt, actions, simulation_times)
end

function run_noisy_grid!(options, manifest, controller, output)
    runtime = configure_runtime!(options, manifest, controller, output)
    paths = String[]
    for replicate in 1:NOISE_REPLICATES
        for case in manifest[:cases]
            case_index = Int(valueof(case, :index))
            case_id = string(valueof(case, :case_id))
            seed = noise_seed(options.protocol, options.noise_level, replicate, case_index)
            path = episode_output_path(output, options.noise_level, replicate, case_id)
            if episode_is_current(path, options, manifest, replicate, case)
                println("Skipping complete episode: r$(lpad(replicate, 2, '0')) / $case_id")
            else
                println("Running $(options.protocol)/$(options.controller)/$(options.noise_level): r$(lpad(replicate, 2, '0')) / $case_id / seed=$seed")
                episode = Base.invokelatest(run_noisy_episode, options, manifest, runtime, case, seed)
                save_episode(path, options, manifest, controller, case, episode; replicate, seed)
            end
            push!(paths, abspath(path))
        end
    end
    return paths
end

function collect_summaries(paths)
    return [begin
        loaded = JLD2.load(path)
        (
            path = abspath(path),
            replicate = Int(loaded["replicate"]),
            noise_seed = haskey(loaded, "noise_seed") ? loaded["noise_seed"] : nothing,
            case_index = Int(loaded["case_index"]),
            case_id = string(loaded["case_id"]),
            reward_sum = Float64(loaded["reward_sum"]),
            mean_reward = Float64(loaded["mean_reward"]),
            sum_state_nusselt = Float64(loaded["sum_state_nusselt"]),
            mean_state_nusselt = Float64(loaded["mean_state_nusselt"]),
            action_saturation_fraction = Float64(loaded["action_saturation_fraction"]),
        )
    end for path in paths]
end

function main(arguments = ARGS)
    options = parse_options(arguments)
    isnothing(options) && return nothing
    manifest = load_protocol_manifest(options.manifest_path)
    controller = validate_manifest(options, manifest)
    output = output_directory(options, manifest)
    status = status_path(output)
    result = result_path(output)
    if status_is_complete(status, options, manifest[:manifest_fingerprint]) && isfile(result)
        println("Noise-Study worker already complete: $result")
        return result
    end
    if isfile(status) && !options.retry_failed
        previous = JLD2.load(status)
        Symbol(previous["state"]) === :failed && error(
            "Worker previously failed. Pass --retry-failed after inspecting: $status",
        )
    end
    mkpath(output)
    write_worker_status!(status, options, manifest; state = :running, started_at = string(Dates.now(Dates.UTC)))
    try
        paths = options.noise_level == 0 ?
            import_clean_baseline!(options, manifest, controller, output) :
            run_noisy_grid!(options, manifest, controller, output)
        expected = options.noise_level == 0 ? length(manifest[:cases]) : NOISE_REPLICATES * length(manifest[:cases])
        length(paths) == expected || error("Expected $expected episode paths, got $(length(paths)).")
        all(isfile, paths) || error("At least one expected episode file is missing.")
        summaries = collect_summaries(paths)
        atomic_save(
            result;
            schema_version = NOISE_SCHEMA_VERSION,
            experiment = :package10_sensor_noise_worker,
            experiment_id = string(manifest[:experiment_id]),
            protocol = options.protocol,
            controller = options.controller,
            controller_id = string(controller[:controller_id]),
            configuration = string(controller[:configuration]),
            noise_level = options.noise_level,
            noise_model = :additive_iid_zero_mean_gaussian,
            replicate_count = options.noise_level == 0 ? 0 : NOISE_REPLICATES,
            case_count = length(manifest[:cases]),
            episode_count = length(paths),
            episode_paths = paths,
            summaries,
            manifest_path = options.manifest_path,
            manifest_sha256 = file_sha256(options.manifest_path),
            manifest_fingerprint = string(manifest[:manifest_fingerprint]),
            completed_at = string(Dates.now(Dates.UTC)),
        )
        write_worker_status!(
            status,
            options,
            manifest;
            state = :complete,
            result_path = abspath(result),
            episode_count = length(paths),
            completed_at = string(Dates.now(Dates.UTC)),
        )
        println("Noise-Study worker complete: $result")
        return result
    catch error_value
        message = sprint(showerror, error_value, catch_backtrace())
        write_worker_status!(status, options, manifest; state = :failed, message)
        rethrow()
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
