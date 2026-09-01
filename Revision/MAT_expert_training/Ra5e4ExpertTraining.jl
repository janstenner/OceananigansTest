# Included inside MATExpertTraining. The Ra=5e4 path deliberately reuses the
# established atomic I/O, locking, hook restoration, one-episode training, and
# compact-expert helpers without changing the existing Ra=1e4 protocols.

ra5e4_protocol_directory(results_directory) = joinpath(results_directory, "varying")
ra5e4_manifest_path(results_directory) = joinpath(results_directory, "experiment_manifest.jld2")
ra5e4_stop_signal_path(results_directory) = joinpath(
    ra5e4_protocol_directory(results_directory),
    "stop_signal.jld2",
)
ra5e4_best_checkpoint_path(results_directory) = joinpath(
    ra5e4_protocol_directory(results_directory),
    "best_so_far.jld2",
)
ra5e4_expert_path(results_directory) = joinpath(
    ra5e4_protocol_directory(results_directory),
    "expert.jld2",
)
ra5e4_worker_directory(results_directory, candidate) = joinpath(
    results_directory,
    "runs",
    candidate.run_id,
)
ra5e4_resume_path(results_directory, candidate) = joinpath(
    ra5e4_worker_directory(results_directory, candidate),
    "resume",
    "latest.jld2",
)
ra5e4_final_path(results_directory, candidate) = joinpath(
    ra5e4_worker_directory(results_directory, candidate),
    "final.jld2",
)
ra5e4_failure_path(results_directory, candidate) = joinpath(
    ra5e4_worker_directory(results_directory, candidate),
    "failure.jld2",
)

function normalize_ra5e4_stop_mode(value)
    mode = Symbol(lowercase(replace(string(value), '-' => '_')))
    mode in (:episodes, :threshold) || throw(ArgumentError(
        "Ra=5e4 stop mode must be 'episodes' or 'threshold', got '$value'.",
    ))
    return mode
end

function ra5e4_candidates(master_seed::Integer, run_count::Integer)
    run_count > 0 || throw(ArgumentError("run_count must be positive."))
    rng = StableRNG(Int(master_seed))
    used = Set{Int}()
    draw_seed() = begin
        value = rand(rng, 1:2_000_000_000)
        while value in used
            value = rand(rng, 1:2_000_000_000)
        end
        push!(used, value)
        value
    end
    width = max(2, ndigits(run_count))
    return [
        begin
            run_seed = draw_seed()
            ic_seed = draw_seed()
            (
                run_index = index,
                run_id = "ra5e4_r$(lpad(string(index), width, '0'))_" *
                         string(run_seed; base = 16, pad = 8),
                run_seed,
                ic_seed,
            )
        end for index in 1:run_count
    ]
end

function ra5e4_dict_value(mapping, key::Symbol)
    haskey(mapping, key) && return mapping[key]
    string_key = string(key)
    haskey(mapping, string_key) && return mapping[string_key]
    error("Missing '$key' in Ra=5e4 corpus metadata.")
end

function validate_ra5e4_corpus(path::AbstractString = RA5E4_CORPUS_PATH)
    isfile(path) || error(
        "Missing Ra=5e4 Varying-IC corpus: $path. Finish corpus generation before launching training.",
    )
    JLD2.jldopen(path, "r") do file
        haskey(file, "corpus") || error("Ra=5e4 corpus has no 'corpus' entry: $path")
        haskey(file, "simulation_config") || error(
            "Ra=5e4 corpus has no 'simulation_config' entry: $path",
        )
        config = read(file, "simulation_config")
        observed_ra = Float64(ra5e4_dict_value(config, :Ra))
        observed_ra == RA5E4_RAYLEIGH || error(
            "Ra=5e4 corpus reports Ra=$observed_ra instead of $RA5E4_RAYLEIGH.",
        )
        corpus = read(file, "corpus")
        for split in (:train, :validation, :test)
            records = ra5e4_dict_value(corpus, split)
            isempty(records) && error("Ra=5e4 corpus split '$split' is empty: $path")
        end
    end
    return source_hash(path)
end

function ra5e4_config_fingerprint(;
    master_seed,
    run_count,
    stop_mode,
    episode_limit,
    threshold,
    corpus_sha256,
    corpus_source_sha256,
    run_file_sha256,
    candidates,
)
    candidate_text = join(
        (
            "$(candidate.run_index):$(candidate.run_id):$(candidate.run_seed):$(candidate.ic_seed)"
            for candidate in candidates
        ),
        ";",
    )
    fields = (
        "schema=$(RA5E4_SCHEMA_VERSION)",
        "protocol=$(RA5E4_PROTOCOL)",
        "ra=$(RA5E4_RAYLEIGH)",
        "window=$(RA5E4_WINDOW)",
        "master_seed=$(Int(master_seed))",
        "run_count=$(Int(run_count))",
        "stop_mode=$(stop_mode)",
        "episode_limit=$(Int(episode_limit))",
        "threshold=$(@sprintf("%.17g", Float64(threshold)))",
        "corpus=$corpus_sha256",
        "corpus_source=$corpus_source_sha256",
        "run_file=$run_file_sha256",
        "candidates=$candidate_text",
    )
    return bytes2hex(SHA.sha256(join(fields, "|")))
end

function build_ra5e4_experiment(;
    master_seed::Integer = DEFAULT_RA5E4_MASTER_SEED,
    run_count::Integer,
    stop_mode,
    episode_limit::Integer = 0,
    threshold::Real = 0.0,
    preview::Bool = false,
)
    mode = normalize_ra5e4_stop_mode(stop_mode)
    run_count > 0 || throw(ArgumentError("run_count must be positive."))
    if mode === :episodes
        episode_limit >= RA5E4_WINDOW || throw(ArgumentError(
            "The fixed episode limit must be at least $RA5E4_WINDOW so a rolling-100 best candidate exists.",
        ))
        normalized_episode_limit = Int(episode_limit)
        normalized_threshold = 0.0
    else
        episode_limit == 0 || throw(ArgumentError(
            "episode_limit must be zero in threshold mode.",
        ))
        isfinite(threshold) || throw(ArgumentError("threshold must be finite."))
        normalized_episode_limit = 0
        normalized_threshold = Float64(threshold)
    end

    isfile(RA5E4_RUN_FILE_PATH) || error("Missing Ra=5e4 run file: $RA5E4_RUN_FILE_PATH")
    isfile(RA5E4_CORPUS_SOURCE_PATH) || error(
        "Missing Ra=5e4 corpus source: $RA5E4_CORPUS_SOURCE_PATH",
    )
    corpus_sha = if preview && !isfile(RA5E4_CORPUS_PATH)
        "pending-corpus-generation"
    else
        validate_ra5e4_corpus()
    end
    corpus_source_sha = source_hash(RA5E4_CORPUS_SOURCE_PATH)
    run_file_sha = source_hash(RA5E4_RUN_FILE_PATH)
    candidates = ra5e4_candidates(master_seed, run_count)
    fingerprint = ra5e4_config_fingerprint(
        ;
        master_seed,
        run_count,
        stop_mode = mode,
        episode_limit = normalized_episode_limit,
        threshold = normalized_threshold,
        corpus_sha256 = corpus_sha,
        corpus_source_sha256 = corpus_source_sha,
        run_file_sha256 = run_file_sha,
        candidates,
    )
    return (
        protocol = RA5E4_PROTOCOL,
        rayleigh = RA5E4_RAYLEIGH,
        rolling_window = RA5E4_WINDOW,
        master_seed = Int(master_seed),
        run_count = Int(run_count),
        stop_mode = mode,
        episode_limit = normalized_episode_limit,
        threshold = normalized_threshold,
        corpus_path = abspath(RA5E4_CORPUS_PATH),
        corpus_sha256 = corpus_sha,
        corpus_source_path = abspath(RA5E4_CORPUS_SOURCE_PATH),
        corpus_source_sha256 = corpus_source_sha,
        run_file_path = abspath(RA5E4_RUN_FILE_PATH),
        run_file_sha256 = run_file_sha,
        config_fingerprint = fingerprint,
        candidates,
    )
end

function freeze_ra5e4_experiment!(;
    results_directory::AbstractString = DEFAULT_RA5E4_RESULTS_DIRECTORY,
    master_seed::Integer = DEFAULT_RA5E4_MASTER_SEED,
    run_count::Integer,
    stop_mode,
    episode_limit::Integer = 0,
    threshold::Real = 0.0,
    preview::Bool = false,
)
    experiment = build_ra5e4_experiment(
        ;
        master_seed,
        run_count,
        stop_mode,
        episode_limit,
        threshold,
        preview,
    )
    preview && return experiment
    path = ra5e4_manifest_path(results_directory)
    return with_lock(path * ".lock"; wait_seconds = 60.0) do
        if isfile(path)
            existing = JLD2.load(path, "experiment")
            existing == experiment || error(
                "Existing Ra=5e4 experiment manifest differs from this launch: $path. " *
                "Use the original arguments to resume or choose a new results directory.",
            )
            return existing
        end
        atomic_jldsave(
            path;
            schema_version = RA5E4_SCHEMA_VERSION,
            status = "frozen",
            created_at = string(now()),
            experiment,
        )
        return experiment
    end
end

function ra5e4_candidate(experiment, run_index::Integer)
    matches = [candidate for candidate in experiment.candidates if
               candidate.run_index == Int(run_index)]
    length(matches) == 1 || error("No unique Ra=5e4 candidate for run index $run_index.")
    return only(matches)
end

function include_ra5e4_run_file!(candidate, runtime_directory)
    ENV["REVISION_RUN_SEED"] = string(candidate.run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    mkpath(runtime_directory)
    Base.include(@__MODULE__, RA5E4_RUN_FILE_PATH)
    hook.is_display_on_exit = false
    hook.display_after_episode = false
    return nothing
end

function ra5e4_training_seeds()
    # VaryingICCorpus_Ra5e4.jl has already loaded the corpus for sampling. Use
    # that object instead of briefly loading a second full copy per worker.
    split = ra5e4_dict_value(CORPUS, :train)
    seeds = sort!(Int.(collect(keys(split))))
    isempty(seeds) && error("Ra=5e4 training corpus is empty.")
    return seeds
end

function configure_ra5e4_episode_initializer!(candidate, observed_trace)
    seeds = ra5e4_training_seeds()
    rng = StableRNG(candidate.ic_seed)
    for (index, observed) in enumerate(observed_trace)
        expected = draw_varying_choice(rng, seeds)
        observed == expected || error(
            "Stored Ra=5e4 IC trace differs at episode $index: expected $expected, got $observed.",
        )
    end
    hook.generate_random_init = () -> begin
        choice = draw_varying_choice(rng, seeds)
        result, split, base_seed, mirror, offset = Base.invokelatest(
            generate_random_init;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        )
        observed = (; split, base_seed, mirror, offset)
        observed == choice || error("Ra=5e4 run file did not reproduce planned IC $choice.")
        push!(observed_trace, observed)
        return result
    end
    return nothing
end

function load_ra5e4_training_state(candidate, results_directory, fingerprint)
    path = ra5e4_resume_path(results_directory, candidate)
    if isfile(path)
        return JLD2.jldopen(path, "r") do file
            string(read(file, "status")) == "running" || error("Invalid resume status: $path")
            string(read(file, "run_id")) == candidate.run_id || error("Resume run-ID mismatch.")
            string(read(file, "config_fingerprint")) == fingerprint || error(
                "Resume configuration fingerprint mismatch: $path",
            )
            (
                agent = read(file, "agent"),
                rewards = Float64.(read(file, "rewards")),
                rewards_all_timesteps = Float64.(read(file, "rewards_all_timesteps")),
                rewards_compare = Float64.(read(file, "rewards_compare")),
                errored_episodes = collect(read(file, "errored_episodes")),
                best_reward = Float64(read(file, "best_reward")),
                best_episode = Int(read(file, "best_episode")),
                episodes_completed = Int(read(file, "episodes_completed")),
                initial_condition_trace = normalize_trace(read(file, "initial_condition_trace")),
                prior_elapsed_seconds = Float64(read(file, "elapsed_seconds")),
            )
        end
    end
    return (
        agent,
        rewards = Float64.(hook.rewards),
        rewards_all_timesteps = Float64.(hook.rewards_all_timesteps),
        rewards_compare = Float64.(hook.rewards_compare),
        errored_episodes = collect(hook.errored_episodes),
        best_reward = Float64(hook.bestreward),
        best_episode = Int(hook.bestepisode),
        episodes_completed = 0,
        initial_condition_trace = NamedTuple[],
        prior_elapsed_seconds = 0.0,
    )
end

function save_ra5e4_resume!(candidate, results_directory, experiment,
                            episodes_completed, initial_condition_trace,
                            elapsed_seconds)
    atomic_jldsave(
        ra5e4_resume_path(results_directory, candidate);
        schema_version = RA5E4_SCHEMA_VERSION,
        status = "running",
        protocol = RA5E4_PROTOCOL,
        rayleigh = RA5E4_RAYLEIGH,
        run_index = candidate.run_index,
        run_id = candidate.run_id,
        run_seed = candidate.run_seed,
        ic_seed = candidate.ic_seed,
        config_fingerprint = experiment.config_fingerprint,
        stop_mode = experiment.stop_mode,
        episode_limit = experiment.episode_limit,
        threshold = experiment.threshold,
        rolling_window = RA5E4_WINDOW,
        episodes_completed = Int(episodes_completed),
        rewards = copy(hook.rewards),
        rewards_all_timesteps = copy(hook.rewards_all_timesteps),
        rewards_compare = copy(hook.rewards_compare),
        errored_episodes = copy(hook.errored_episodes),
        best_reward = hook.bestreward,
        best_episode = hook.bestepisode,
        initial_condition_trace = copy(initial_condition_trace),
        elapsed_seconds = Float64(elapsed_seconds),
        updated_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        agent,
    )
end

ra5e4_criterion_value(rewards) = length(rewards) < RA5E4_WINDOW ? NaN :
    mean(@view rewards[(end - RA5E4_WINDOW + 1):end])

function save_ra5e4_best_so_far!(candidate, results_directory, experiment,
                                 episodes_completed, initial_condition_trace,
                                 elapsed_seconds)
    metric = ra5e4_criterion_value(hook.rewards)
    isfinite(metric) || return false
    protocol_directory = ra5e4_protocol_directory(results_directory)
    mkpath(protocol_directory)
    checkpoint_path = ra5e4_best_checkpoint_path(results_directory)
    expert_path = ra5e4_expert_path(results_directory)
    return with_lock(
        joinpath(protocol_directory, ".best_so_far.lock");
        wait_seconds = BEST_LOCK_WAIT_SECONDS,
    ) do
        if isfile(checkpoint_path)
            existing = JLD2.jldopen(checkpoint_path, "r") do file
                string(read(file, "status")) == "best_so_far" || error(
                    "Invalid Ra=5e4 best-so-far status: $checkpoint_path",
                )
                string(read(file, "config_fingerprint")) == experiment.config_fingerprint ||
                    error("Ra=5e4 best-so-far configuration mismatch.")
                Float64(read(file, "criterion_value"))
            end
            if metric <= existing
                compact_expert_checkpoint_valid(expert_path) ||
                    save_compact_expert_from_checkpoint!(checkpoint_path, expert_path)
                return false
            end
        end
        atomic_best_so_far_save(
            checkpoint_path,
            expert_path;
            schema_version = RA5E4_SCHEMA_VERSION,
            status = "best_so_far",
            protocol = RA5E4_PROTOCOL,
            rayleigh = RA5E4_RAYLEIGH,
            run_index = candidate.run_index,
            run_id = candidate.run_id,
            run_seed = candidate.run_seed,
            ic_seed = candidate.ic_seed,
            config_fingerprint = experiment.config_fingerprint,
            selection_basis = "mean_latest_100_completed_episode_rewards",
            rolling_window = RA5E4_WINDOW,
            criterion_value = Float64(metric),
            stop_mode = experiment.stop_mode,
            threshold = experiment.threshold,
            threshold_reached = experiment.stop_mode === :threshold &&
                metric > experiment.threshold,
            episodes_completed = Int(episodes_completed),
            latest_episode_reward = Float64(last(hook.rewards)),
            rewards = copy(hook.rewards),
            rewards_all_timesteps = copy(hook.rewards_all_timesteps),
            rewards_compare = copy(hook.rewards_compare),
            errored_episodes = copy(hook.errored_episodes),
            best_reward = hook.bestreward,
            best_episode = hook.bestepisode,
            initial_condition_trace = copy(initial_condition_trace),
            elapsed_seconds = Float64(elapsed_seconds),
            updated_at = string(now()),
            hostname = gethostname(),
            julia_version = string(VERSION),
            agent,
        )
        return true
    end
end

function read_ra5e4_stop_signal(results_directory, fingerprint)
    path = ra5e4_stop_signal_path(results_directory)
    isfile(path) || return nothing
    signal = JLD2.load(path)
    string(signal["config_fingerprint"]) == fingerprint || error(
        "Ra=5e4 stop-signal configuration mismatch: $path",
    )
    return signal
end

function publish_ra5e4_threshold!(candidate, results_directory, experiment,
                                  metric, episodes_completed)
    directory = ra5e4_protocol_directory(results_directory)
    mkpath(directory)
    return with_lock(joinpath(directory, ".candidate.lock"); wait_seconds = 60.0) do
        existing = read_ra5e4_stop_signal(results_directory, experiment.config_fingerprint)
        !isnothing(existing) && return false
        atomic_jldsave(
            ra5e4_stop_signal_path(results_directory);
            schema_version = RA5E4_SCHEMA_VERSION,
            status = "stop_requested",
            selection_mode = "rolling_100_threshold",
            protocol = RA5E4_PROTOCOL,
            rayleigh = RA5E4_RAYLEIGH,
            config_fingerprint = experiment.config_fingerprint,
            winner_run_index = candidate.run_index,
            winner_run_id = candidate.run_id,
            run_seed = candidate.run_seed,
            ic_seed = candidate.ic_seed,
            rolling_window = RA5E4_WINDOW,
            criterion_value = Float64(metric),
            threshold = experiment.threshold,
            episodes_completed = Int(episodes_completed),
            requested_at = string(now()),
        )
        return true
    end
end

function ra5e4_final_complete(path, candidate, fingerprint)
    isfile(path) || return false
    return try
        JLD2.jldopen(path, "r") do file
            string(read(file, "status")) == "complete" &&
            string(read(file, "run_id")) == candidate.run_id &&
            string(read(file, "config_fingerprint")) == fingerprint
        end
    catch
        false
    end
end

function save_ra5e4_final!(candidate, results_directory, experiment,
                           episodes_completed, initial_condition_trace,
                           elapsed_seconds, stop_signal)
    threshold_winner = experiment.stop_mode === :threshold &&
        !isnothing(stop_signal) && string(stop_signal["winner_run_id"]) == candidate.run_id
    completion_reason = experiment.stop_mode === :episodes ? "episode_limit_reached" :
        (threshold_winner ? "threshold_reached" : "stopped_by_threshold_winner")
    atomic_jldsave(
        ra5e4_final_path(results_directory, candidate);
        schema_version = RA5E4_SCHEMA_VERSION,
        status = "complete",
        completion_reason,
        is_threshold_winner = threshold_winner,
        protocol = RA5E4_PROTOCOL,
        rayleigh = RA5E4_RAYLEIGH,
        run_index = candidate.run_index,
        run_id = candidate.run_id,
        run_seed = candidate.run_seed,
        ic_seed = candidate.ic_seed,
        config_fingerprint = experiment.config_fingerprint,
        stop_mode = experiment.stop_mode,
        episode_limit = experiment.episode_limit,
        threshold = experiment.threshold,
        rolling_window = RA5E4_WINDOW,
        episodes_completed = Int(episodes_completed),
        final_criterion_value = ra5e4_criterion_value(hook.rewards),
        winner_run_id = isnothing(stop_signal) ? "" : string(stop_signal["winner_run_id"]),
        rewards = copy(hook.rewards),
        rewards_all_timesteps = copy(hook.rewards_all_timesteps),
        rewards_compare = copy(hook.rewards_compare),
        errored_episodes = copy(hook.errored_episodes),
        best_reward = hook.bestreward,
        best_episode = hook.bestepisode,
        initial_condition_trace = copy(initial_condition_trace),
        elapsed_seconds = Float64(elapsed_seconds),
        completed_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        agent,
    )
end

function save_ra5e4_failure!(candidate, results_directory, experiment, message)
    atomic_jldsave(
        ra5e4_failure_path(results_directory, candidate);
        schema_version = RA5E4_SCHEMA_VERSION,
        status = "failed",
        protocol = RA5E4_PROTOCOL,
        rayleigh = RA5E4_RAYLEIGH,
        run_index = candidate.run_index,
        run_id = candidate.run_id,
        config_fingerprint = experiment.config_fingerprint,
        failed_at = string(now()),
        error_message = message,
    )
end

function run_ra5e4_training_worker(;
    run_index::Integer,
    results_directory::AbstractString = DEFAULT_RA5E4_RESULTS_DIRECTORY,
    master_seed::Integer = DEFAULT_RA5E4_MASTER_SEED,
    run_count::Integer,
    stop_mode,
    episode_limit::Integer = 0,
    threshold::Real = 0.0,
)
    experiment = freeze_ra5e4_experiment!(
        ;
        results_directory,
        master_seed,
        run_count,
        stop_mode,
        episode_limit,
        threshold,
    )
    candidate = ra5e4_candidate(experiment, run_index)
    output_directory = ra5e4_worker_directory(results_directory, candidate)
    mkpath(output_directory)
    lock = acquire_lock(joinpath(output_directory, ".worker.lock"))
    try
        final_path = ra5e4_final_path(results_directory, candidate)
        if ra5e4_final_complete(final_path, candidate, experiment.config_fingerprint)
            println("Complete Ra=5e4 result already exists; skipping $(candidate.run_id).")
            return final_path
        end

        include_ra5e4_run_file!(candidate, joinpath(output_directory, "runtime"))
        state = load_ra5e4_training_state(
            candidate,
            results_directory,
            experiment.config_fingerprint,
        )
        global agent = state.agent
        restore_hook!(state)
        initial_condition_trace = copy(state.initial_condition_trace)
        configure_ra5e4_episode_initializer!(candidate, initial_condition_trace)
        episodes_completed = state.episodes_completed
        episodes_completed == length(hook.rewards) || error(
            "Ra=5e4 episode count and stored reward count differ.",
        )
        length(initial_condition_trace) == episodes_completed || error(
            "Ra=5e4 IC trace and episode count differ.",
        )
        elapsed_seconds = state.prior_elapsed_seconds

        hook(PRE_EXPERIMENT_STAGE, agent, env)
        agent(PRE_EXPERIMENT_STAGE, env)
        save_ra5e4_resume!(
            candidate,
            results_directory,
            experiment,
            episodes_completed,
            initial_condition_trace,
            elapsed_seconds,
        )
        initial_best_updated = save_ra5e4_best_so_far!(
            candidate,
            results_directory,
            experiment,
            episodes_completed,
            initial_condition_trace,
            elapsed_seconds,
        )
        metric = ra5e4_criterion_value(hook.rewards)
        stop_signal = experiment.stop_mode === :threshold ?
            read_ra5e4_stop_signal(results_directory, experiment.config_fingerprint) : nothing
        if experiment.stop_mode === :threshold && isnothing(stop_signal) &&
           isfinite(metric) && metric > experiment.threshold
            won = publish_ra5e4_threshold!(
                candidate,
                results_directory,
                experiment,
                metric,
                episodes_completed,
            )
            won && println("Published Ra=5e4 threshold stop from resumed state.")
            stop_signal = read_ra5e4_stop_signal(
                results_directory,
                experiment.config_fingerprint,
            )
        end
        println(
            "Starting Ra=5e4 $(candidate.run_id) from episode $episodes_completed; " *
            "stop_mode=$(experiment.stop_mode), " *
            (experiment.stop_mode === :episodes ?
                "episode_limit=$(experiment.episode_limit)" :
                "threshold=$(experiment.threshold) after rolling window $(RA5E4_WINDOW)") *
            ", initial_criterion=$metric, initial_global_best_updated=$initial_best_updated.",
        )
        flush(stdout)

        while true
            experiment.stop_mode === :episodes &&
                episodes_completed >= experiment.episode_limit && break
            experiment.stop_mode === :threshold && !isnothing(stop_signal) && break

            episode_elapsed = @elapsed episode_reward = Base.invokelatest(train_one_episode!)
            elapsed_seconds += episode_elapsed
            episodes_completed += 1
            length(initial_condition_trace) == episodes_completed || error(
                "Ra=5e4 IC trace and episode count differ after training.",
            )
            save_ra5e4_resume!(
                candidate,
                results_directory,
                experiment,
                episodes_completed,
                initial_condition_trace,
                elapsed_seconds,
            )
            metric = ra5e4_criterion_value(hook.rewards)
            best_updated = save_ra5e4_best_so_far!(
                candidate,
                results_directory,
                experiment,
                episodes_completed,
                initial_condition_trace,
                elapsed_seconds,
            )
            reached = experiment.stop_mode === :threshold && isfinite(metric) &&
                metric > experiment.threshold
            @printf(
                "[%s] protocol=varying_ra5e4 run=%d/%d id=%s episode=%d reward=%.6f rolling_100=%.6f stop_mode=%s target=%s reached=%s global_best_updated=%s\n",
                now(),
                candidate.run_index,
                experiment.run_count,
                candidate.run_id,
                episodes_completed,
                episode_reward,
                metric,
                experiment.stop_mode,
                experiment.stop_mode === :episodes ? string(experiment.episode_limit) :
                    string(experiment.threshold),
                reached,
                best_updated,
            )
            flush(stdout)
            if reached
                won = publish_ra5e4_threshold!(
                    candidate,
                    results_directory,
                    experiment,
                    metric,
                    episodes_completed,
                )
                won && println("Published Ra=5e4 threshold winner $(candidate.run_id).")
            end
            stop_signal = experiment.stop_mode === :threshold ?
                read_ra5e4_stop_signal(results_directory, experiment.config_fingerprint) : nothing
        end

        hook(POST_EXPERIMENT_STAGE, agent, env)
        save_ra5e4_final!(
            candidate,
            results_directory,
            experiment,
            episodes_completed,
            initial_condition_trace,
            elapsed_seconds,
            stop_signal,
        )
        failure_path = ra5e4_failure_path(results_directory, candidate)
        isfile(failure_path) && rm(failure_path; force = true)
        println(
            "Saved Ra=5e4 final checkpoint for $(candidate.run_id): " *
            "episodes=$episodes_completed, rolling_100=$(ra5e4_criterion_value(hook.rewards)).",
        )
        flush(stdout)
        return final_path
    catch error_value
        message = sprint(showerror, error_value, catch_backtrace())
        save_ra5e4_failure!(candidate, results_directory, experiment, message)
        rethrow(error_value)
    finally
        isdir(lock) && rm(lock; recursive = true, force = true)
    end
end


