module MATExpertTraining

using Dates
using Flux
using JLD2
using PlotlyJS
using Printf
using Random
using RL
using SHA
using Sockets
using StableRNGs
using Statistics
using UUIDs

export CANDIDATES, DEFAULT_RESULTS_DIRECTORY, DEFAULT_SOURCE_RESULTS_DIRECTORY,
       FIXED_THRESHOLD, VARYING_THRESHOLD, freeze_selection_manifest!,
       run_training_worker, run_test_protocol_worker, candidate_ready,
       protocol_ready_for_test, test_complete, source_checkpoint_path,
       publish_distillation_experts!, experts_published

const SCHEMA_VERSION = 1
const REVISION_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
const PROJECT_ROOT = normpath(joinpath(REVISION_DIRECTORY, ".."))
const DEFAULT_RESULTS_DIRECTORY = joinpath(@__DIR__, "results")
const DEFAULT_SOURCE_RESULTS_DIRECTORY = joinpath(
    REVISION_DIRECTORY,
    "MAT_IPPO_Comparison",
    "results",
)
const DEFAULT_DISTILLATION_EXPERT_DIRECTORY = joinpath(
    REVISION_DIRECTORY,
    "Expert_Apprentice_Distillation",
    "experts",
)
const CORPUS_PATH = joinpath(
    REVISION_DIRECTORY,
    "VaryingIC_Corpus",
    "varying_ic_corpus.jld2",
)
const ORIGINAL_EPISODES = Dict(:fixed => 2_000, :varying => 4_000)
const FIXED_THRESHOLD = -555.0
const VARYING_THRESHOLD = -610.0
const VARYING_WINDOW = 100
const TEST_EPISODE_STEPS = 200
const PROTOCOLS = (:fixed, :varying)
# Validation returns can differ in their last Float32-derived digits across
# Julia/CUDA/hardware environments. This tolerance is still orders of magnitude
# below the smallest score gap in either frozen ranking.
const VALIDATION_SCORE_ATOL = 1e-5

const CANDIDATES = (
    (
        protocol = :fixed,
        rank = 1,
        run_id = "seed_ce0b5b582dda8eff",
        validation_score = -588.662153945814,
        run_seed = 686_791_604,
        ic_seed = 493_568_598,
        origin = :imported_package3,
    ),
    (
        protocol = :fixed,
        rank = 2,
        run_id = "seed_1f82af6b2a587ef6",
        validation_score = -590.433882645501,
        run_seed = 1_241_319_044,
        ic_seed = 83_126_739,
        origin = :imported_package3,
    ),
    (
        protocol = :fixed,
        rank = 3,
        run_id = "seed_a0c258998e2632fd",
        validation_score = -592.4416938887673,
        run_seed = 740_592_503,
        ic_seed = 1_545_505_641,
        origin = :generated,
    ),
    (
        protocol = :fixed,
        rank = 4,
        run_id = "seed_41296dceb8db78ce",
        validation_score = -593.5512835444865,
        run_seed = 874_617_006,
        ic_seed = 579_508_289,
        origin = :generated,
    ),
    (
        protocol = :fixed,
        rank = 5,
        run_id = "seed_500415d5315402cd",
        validation_score = -594.1152706696137,
        run_seed = 633_980_338,
        ic_seed = 386_963_254,
        origin = :generated,
    ),
    (
        protocol = :fixed,
        rank = 6,
        run_id = "seed_b983871a14d50ecb",
        validation_score = -594.4241241430935,
        run_seed = 530_442_171,
        ic_seed = 122_357_054,
        origin = :generated,
    ),
    (
        protocol = :fixed,
        rank = 7,
        run_id = "seed_e72249f3ea1fa410",
        validation_score = -600.5881163654836,
        run_seed = 1_422_047_759,
        ic_seed = 627_765_402,
        origin = :imported_package3,
    ),
    (
        protocol = :fixed,
        rank = 8,
        run_id = "seed_f6984b8779f1349c",
        validation_score = -601.1631889210348,
        run_seed = 1_863_733_649,
        ic_seed = 973_985_366,
        origin = :generated,
    ),
    (
        protocol = :fixed,
        rank = 9,
        run_id = "seed_3a2d3a2f3341b412",
        validation_score = -615.0248757752528,
        run_seed = 319_470_045,
        ic_seed = 1_788_533_232,
        origin = :imported_package3,
    ),
    (
        protocol = :fixed,
        rank = 10,
        run_id = "seed_92b79c49251eb7a2",
        validation_score = -703.1870046063658,
        run_seed = 1_987_317_423,
        ic_seed = 60_237_239,
        origin = :imported_package3,
    ),
    (
        protocol = :varying,
        rank = 1,
        run_id = "seed_92b79c49251eb7a2",
        validation_score = -612.4201562246841,
        run_seed = 1_987_317_423,
        ic_seed = 60_237_239,
        origin = :imported_package3,
    ),
    (
        protocol = :varying,
        rank = 2,
        run_id = "seed_a0c258998e2632fd",
        validation_score = -617.5392566808855,
        run_seed = 740_592_503,
        ic_seed = 1_545_505_641,
        origin = :generated,
    ),
    (
        protocol = :varying,
        rank = 3,
        run_id = "seed_b983871a14d50ecb",
        validation_score = -619.575374499492,
        run_seed = 530_442_171,
        ic_seed = 122_357_054,
        origin = :generated,
    ),
    (
        protocol = :varying,
        rank = 4,
        run_id = "seed_1f82af6b2a587ef6",
        validation_score = -619.9315436427753,
        run_seed = 1_241_319_044,
        ic_seed = 83_126_739,
        origin = :imported_package3,
    ),
    (
        protocol = :varying,
        rank = 5,
        run_id = "seed_e72249f3ea1fa410",
        validation_score = -620.4323211994139,
        run_seed = 1_422_047_759,
        ic_seed = 627_765_402,
        origin = :imported_package3,
    ),
    (
        protocol = :varying,
        rank = 6,
        run_id = "seed_ce0b5b582dda8eff",
        validation_score = -621.221352270041,
        run_seed = 686_791_604,
        ic_seed = 493_568_598,
        origin = :imported_package3,
    ),
    (
        protocol = :varying,
        rank = 7,
        run_id = "seed_3a2d3a2f3341b412",
        validation_score = -635.0695292787293,
        run_seed = 319_470_045,
        ic_seed = 1_788_533_232,
        origin = :imported_package3,
    ),
    (
        protocol = :varying,
        rank = 8,
        run_id = "seed_500415d5315402cd",
        validation_score = -651.7820854079739,
        run_seed = 633_980_338,
        ic_seed = 386_963_254,
        origin = :generated,
    ),
    (
        protocol = :varying,
        rank = 9,
        run_id = "seed_41296dceb8db78ce",
        validation_score = -654.0193289454812,
        run_seed = 874_617_006,
        ic_seed = 579_508_289,
        origin = :generated,
    ),
    (
        protocol = :varying,
        rank = 10,
        run_id = "seed_f6984b8779f1349c",
        validation_score = -715.0087688479787,
        run_seed = 1_863_733_649,
        ic_seed = 973_985_366,
        origin = :generated,
    ),
)

# Populated by exactly one dynamically included Revision run file per process.
agent = nothing
hook = nothing
env = nothing
model = nothing
function generate_random_init end

normalize_protocol(value) = begin
    protocol = Symbol(lowercase(string(value)))
    protocol in PROTOCOLS || throw(ArgumentError("Unknown protocol '$value'."))
    protocol
end

selected_protocols(value) = begin
    selection = Symbol(lowercase(string(value)))
    selection === :all ? collect(PROTOCOLS) : [normalize_protocol(selection)]
end

source_hash(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function atomic_jldsave(path::AbstractString; values...)
    mkpath(dirname(path))
    temporary = joinpath(dirname(path), ".$(basename(path)).$(uuid4()).tmp")
    try
        JLD2.jldsave(temporary; values...)
        mv(temporary, path; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return path
end

function acquire_lock(path::AbstractString; wait_seconds::Real = 0.0)
    deadline = time() + Float64(wait_seconds)
    while true
        try
            mkdir(path)
            return path
        catch error_value
            if isdir(path) && time() < deadline
                sleep(0.1)
                continue
            end
            isdir(path) && error("Lock already exists: $path")
            rethrow(error_value)
        end
    end
end

function with_lock(f::Function, path::AbstractString; wait_seconds::Real = 0.0)
    lock = acquire_lock(path; wait_seconds)
    try
        return f()
    finally
        isdir(lock) && rm(lock; recursive = true, force = true)
    end
end

source_checkpoint_path(source_results_directory, candidate) = joinpath(
    source_results_directory,
    "runs",
    candidate.run_id,
    string(candidate.protocol),
    "mat.jld2",
)

source_validation_path(source_results_directory, candidate) = joinpath(
    source_results_directory,
    "runs",
    candidate.run_id,
    string(candidate.protocol),
    "mat_validation.jld2",
)

worker_directory(results_directory, candidate) = joinpath(
    results_directory,
    "runs",
    string(candidate.protocol),
    "rank_$(lpad(string(candidate.rank), 2, '0'))_$(candidate.run_id)",
)

resume_path(results_directory, candidate) = joinpath(
    worker_directory(results_directory, candidate),
    "resume",
    "latest.jld2",
)

final_path(results_directory, candidate) = joinpath(
    worker_directory(results_directory, candidate),
    "final.jld2",
)

stop_signal_path(results_directory, protocol) = joinpath(
    results_directory,
    string(protocol),
    "stop_signal.jld2",
)

candidate_manifest_path(results_directory, protocol) = joinpath(
    results_directory,
    string(protocol),
    "candidate.jld2",
)

selection_manifest_path(results_directory) = joinpath(results_directory, "selection_manifest.jld2")

function candidate_for(protocol, run_id)
    protocol = normalize_protocol(protocol)
    matches = [candidate for candidate in CANDIDATES if
               candidate.protocol === protocol && candidate.run_id == string(run_id)]
    length(matches) == 1 || error(
        "Expected one selected candidate for $protocol/$run_id, found $(length(matches)).",
    )
    return only(matches)
end

function read_source_metadata(path)
    return JLD2.jldopen(path, "r") do file
        (
            status = string(read(file, "status")),
            protocol = Symbol(read(file, "protocol")),
            algorithm = Symbol(read(file, "algorithm")),
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
            episodes_completed = Int(read(file, "episodes_completed")),
            rewards = Float64.(read(file, "rewards")),
            rewards_all_timesteps = Float64.(read(file, "rewards_all_timesteps")),
            rewards_compare = Float64.(read(file, "rewards_compare")),
            errored_episodes = collect(read(file, "errored_episodes")),
            best_reward = Float64(read(file, "best_reward")),
            best_episode = Int(read(file, "best_episode")),
            trace = haskey(file, "initial_condition_trace") ?
                normalize_trace(read(file, "initial_condition_trace")) : NamedTuple[],
        )
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

function read_validation_metadata(path)
    return JLD2.jldopen(path, "r") do file
        (
            status = string(read(file, "status")),
            protocol = Symbol(read(file, "protocol")),
            algorithm = Symbol(read(file, "algorithm")),
            run_seed = Int(read(file, "run_seed")),
            ic_seed = Int(read(file, "ic_seed")),
            validation_mean = Float64(read(file, "validation_mean")),
        )
    end
end

function verified_candidate_record(candidate, source_results_directory)
    checkpoint = source_checkpoint_path(source_results_directory, candidate)
    validation = source_validation_path(source_results_directory, candidate)
    isfile(checkpoint) || error("Missing selected checkpoint: $checkpoint")
    isfile(validation) || error("Missing selected validation: $validation")
    source = read_source_metadata(checkpoint)
    score = read_validation_metadata(validation)
    source.status == "complete" || error("Selected checkpoint is not complete: $checkpoint")
    source.protocol === candidate.protocol || error("Selected checkpoint protocol mismatch.")
    source.algorithm === :mat || error("Selected checkpoint is not MAT.")
    source.run_seed == candidate.run_seed || error("Selected checkpoint run-seed mismatch.")
    source.ic_seed == candidate.ic_seed || error("Selected checkpoint IC-seed mismatch.")
    source.episodes_completed == ORIGINAL_EPISODES[candidate.protocol] || error(
        "Selected checkpoint has $(source.episodes_completed) episodes; expected " *
        "$(ORIGINAL_EPISODES[candidate.protocol]).",
    )
    score.status == "complete" || error("Selected validation is not complete: $validation")
    score.protocol === candidate.protocol || error("Selected validation protocol mismatch.")
    score.algorithm === :mat || error("Selected validation is not MAT.")
    score.run_seed == candidate.run_seed || error("Selected validation run-seed mismatch.")
    score.ic_seed == candidate.ic_seed || error("Selected validation IC-seed mismatch.")
    score_delta = score.validation_mean - candidate.validation_score
    isapprox(
        score.validation_mean,
        candidate.validation_score;
        atol = VALIDATION_SCORE_ATOL,
        rtol = 0.0,
    ) || error(
        "Selected validation score $(score.validation_mean) does not match frozen score " *
        "$(candidate.validation_score) within absolute tolerance " *
        "$VALIDATION_SCORE_ATOL (delta=$score_delta).",
    )
    candidate.protocol === :varying && length(source.trace) != ORIGINAL_EPISODES[:varying] &&
        error("Selected Varying checkpoint has an incomplete IC trace.")
    return merge(candidate, (
        observed_validation_score = score.validation_mean,
        validation_score_delta = score_delta,
        checkpoint_path = abspath(checkpoint),
        checkpoint_sha256 = source_hash(checkpoint),
        validation_path = abspath(validation),
        validation_sha256 = source_hash(validation),
    ))
end

function freeze_selection_manifest!(;
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    source_results_directory::AbstractString = DEFAULT_SOURCE_RESULTS_DIRECTORY,
    preview::Bool = false,
)
    records = [verified_candidate_record(candidate, source_results_directory) for candidate in CANDIDATES]
    preview && return records
    path = selection_manifest_path(results_directory)
    with_lock(path * ".lock"; wait_seconds = 60.0) do
        if isfile(path)
            existing = JLD2.load(path, "candidates")
            existing == records || error(
                "Existing selection manifest differs from the frozen twenty candidates: $path",
            )
            return records
        end
        atomic_jldsave(
            path;
            schema_version = SCHEMA_VERSION,
            status = "frozen",
            selection_rule = "all_ten_final_mat_checkpoints_ranked_by_validation_score_per_protocol",
            fixed_threshold = FIXED_THRESHOLD,
            varying_threshold = VARYING_THRESHOLD,
            varying_window = VARYING_WINDOW,
            created_at = string(now()),
            candidates = records,
        )
    end
    return records
end

function run_file_path(protocol)
    filename = protocol === :fixed ? "FixedIC_MAT.jl" : "VaryingIC_MAT.jl"
    return joinpath(REVISION_DIRECTORY, "Run_Files", filename)
end

function include_run_file!(candidate, runtime_directory)
    ENV["REVISION_RUN_SEED"] = string(candidate.run_seed)
    ENV["REVISION_RUN_DIRECTORY"] = runtime_directory
    mkpath(runtime_directory)
    Base.include(@__MODULE__, run_file_path(candidate.protocol))
    hook.is_display_on_exit = false
    hook.display_after_episode = false
    return nothing
end

function restore_hook!(state)
    hook.rewards = Float64.(state.rewards)
    hook.rewards_all_timesteps = Float64.(state.rewards_all_timesteps)
    hook.rewards_compare = Float64.(state.rewards_compare)
    hook.errored_episodes = collect(state.errored_episodes)
    hook.bestreward = Float64(state.best_reward)
    hook.bestepisode = Int(state.best_episode)
    hook.ep = length(hook.rewards) + 1
    hook.reward = 0.0
    hook.is_display_on_exit = false
    hook.display_after_episode = false
    return nothing
end

function training_seeds()
    raw = JLD2.load(CORPUS_PATH, "corpus")
    split = haskey(raw, :train) ? raw[:train] : raw["train"]
    return sort!(Int.(collect(keys(split))))
end

draw_varying_choice(rng, seeds) = (
    split = :train,
    base_seed = rand(rng, seeds),
    mirror = rand(rng, Bool),
    offset = rand(rng, 0:95),
)

function configure_episode_initializer!(candidate, source_trace, continuation_trace)
    if candidate.protocol === :fixed
        hook.generate_random_init = () -> Base.invokelatest(generate_random_init)
        return nothing
    end

    seeds = training_seeds()
    rng = StableRNG(candidate.ic_seed)
    for (index, observed) in enumerate(source_trace)
        expected = draw_varying_choice(rng, seeds)
        observed == expected || error(
            "Stored source IC trace differs at episode $index: expected $expected, got $observed.",
        )
    end
    for (index, observed) in enumerate(continuation_trace)
        expected = draw_varying_choice(rng, seeds)
        observed == expected || error(
            "Stored continuation IC trace differs at additional episode $index.",
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
        observed == choice || error("Run file did not reproduce planned Varying IC $choice.")
        push!(continuation_trace, observed)
        return result
    end
    return nothing
end

function load_training_state(candidate, results_directory, source_results_directory, parent_sha)
    own_resume = resume_path(results_directory, candidate)
    if isfile(own_resume)
        return JLD2.jldopen(own_resume, "r") do file
            string(read(file, "status")) == "running" || error("Invalid resume status: $own_resume")
            string(read(file, "run_id")) == candidate.run_id || error("Resume run-ID mismatch.")
            Symbol(read(file, "protocol")) === candidate.protocol || error("Resume protocol mismatch.")
            string(read(file, "parent_checkpoint_sha256")) == parent_sha ||
                error("Resume parent-checkpoint hash mismatch.")
            (
                agent = read(file, "agent"),
                rewards = Float64.(read(file, "rewards")),
                rewards_all_timesteps = Float64.(read(file, "rewards_all_timesteps")),
                rewards_compare = Float64.(read(file, "rewards_compare")),
                errored_episodes = collect(read(file, "errored_episodes")),
                best_reward = Float64(read(file, "best_reward")),
                best_episode = Int(read(file, "best_episode")),
                additional_episodes = Int(read(file, "additional_episodes")),
                continuation_trace = normalize_trace(read(file, "continuation_trace")),
                prior_elapsed_seconds = Float64(read(file, "elapsed_seconds")),
            )
        end
    end

    source_path = source_checkpoint_path(source_results_directory, candidate)
    source = read_source_metadata(source_path)
    return (
        agent = JLD2.load(source_path, "agent"),
        rewards = source.rewards,
        rewards_all_timesteps = source.rewards_all_timesteps,
        rewards_compare = source.rewards_compare,
        errored_episodes = source.errored_episodes,
        best_reward = source.best_reward,
        best_episode = source.best_episode,
        additional_episodes = 0,
        continuation_trace = NamedTuple[],
        prior_elapsed_seconds = 0.0,
    )
end

function save_resume!(candidate, results_directory, parent_path, parent_sha,
                      additional_episodes, continuation_trace, elapsed_seconds)
    atomic_jldsave(
        resume_path(results_directory, candidate);
        schema_version = SCHEMA_VERSION,
        status = "running",
        protocol = candidate.protocol,
        rank = candidate.rank,
        run_id = candidate.run_id,
        run_seed = candidate.run_seed,
        ic_seed = candidate.ic_seed,
        parent_checkpoint_path = abspath(parent_path),
        parent_checkpoint_sha256 = parent_sha,
        original_episodes = ORIGINAL_EPISODES[candidate.protocol],
        additional_episodes = additional_episodes,
        total_episodes = length(hook.rewards),
        rewards = copy(hook.rewards),
        rewards_all_timesteps = copy(hook.rewards_all_timesteps),
        rewards_compare = copy(hook.rewards_compare),
        errored_episodes = copy(hook.errored_episodes),
        best_reward = hook.bestreward,
        best_episode = hook.bestepisode,
        continuation_trace = copy(continuation_trace),
        elapsed_seconds = Float64(elapsed_seconds),
        updated_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        agent = agent,
    )
end

criterion_value(protocol, rewards) = protocol === :fixed ? last(rewards) :
    (length(rewards) >= VARYING_WINDOW ? mean(@view rewards[(end - VARYING_WINDOW + 1):end]) : NaN)

criterion_reached(protocol, rewards) = protocol === :fixed ?
    (!isempty(rewards) && last(rewards) > FIXED_THRESHOLD) :
    (length(rewards) >= VARYING_WINDOW &&
     mean(@view rewards[(end - VARYING_WINDOW + 1):end]) > VARYING_THRESHOLD)

function read_stop_signal(results_directory, protocol)
    path = stop_signal_path(results_directory, protocol)
    isfile(path) || return nothing
    return JLD2.load(path)
end

function publish_candidate!(candidate, results_directory, metric_value, additional_episodes)
    protocol_directory = joinpath(results_directory, string(candidate.protocol))
    mkpath(protocol_directory)
    lock_path = joinpath(protocol_directory, ".candidate.lock")
    return with_lock(lock_path; wait_seconds = 60.0) do
        existing = read_stop_signal(results_directory, candidate.protocol)
        !isnothing(existing) && return false
        final_checkpoint = final_path(results_directory, candidate)
        values = (
            schema_version = SCHEMA_VERSION,
            status = "stop_requested",
            protocol = candidate.protocol,
            winner_run_id = candidate.run_id,
            winner_rank = candidate.rank,
            run_seed = candidate.run_seed,
            ic_seed = candidate.ic_seed,
            criterion = candidate.protocol === :fixed ?
                "episode_reward_gt_-555" : "mean_latest_100_episode_rewards_gt_-610",
            criterion_value = Float64(metric_value),
            original_episodes = ORIGINAL_EPISODES[candidate.protocol],
            additional_episodes = Int(additional_episodes),
            total_episodes = ORIGINAL_EPISODES[candidate.protocol] + Int(additional_episodes),
            final_checkpoint_path = abspath(final_checkpoint),
            requested_at = string(now()),
        )
        atomic_jldsave(stop_signal_path(results_directory, candidate.protocol); values...)
        atomic_jldsave(candidate_manifest_path(results_directory, candidate.protocol); values...)
        return true
    end
end

function train_one_episode!()
    Base.invokelatest(reset!, env)
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
    return hook.rewards[end]
end

function final_result_complete(path, candidate, parent_sha)
    isfile(path) || return false
    return try
        JLD2.jldopen(path, "r") do file
            string(read(file, "status")) == "complete" &&
            string(read(file, "run_id")) == candidate.run_id &&
            Symbol(read(file, "protocol")) === candidate.protocol &&
            string(read(file, "parent_checkpoint_sha256")) == parent_sha
        end
    catch
        false
    end
end

function save_final!(candidate, results_directory, parent_path, parent_sha,
                     additional_episodes, continuation_trace, elapsed_seconds, stop_signal)
    winner = string(stop_signal["winner_run_id"]) == candidate.run_id
    atomic_jldsave(
        final_path(results_directory, candidate);
        schema_version = SCHEMA_VERSION,
        status = "complete",
        completion_reason = winner ? "criterion_reached" : "stopped_by_protocol_winner",
        is_protocol_winner = winner,
        protocol = candidate.protocol,
        rank = candidate.rank,
        run_id = candidate.run_id,
        run_seed = candidate.run_seed,
        ic_seed = candidate.ic_seed,
        parent_checkpoint_path = abspath(parent_path),
        parent_checkpoint_sha256 = parent_sha,
        original_episodes = ORIGINAL_EPISODES[candidate.protocol],
        additional_episodes = additional_episodes,
        total_episodes = length(hook.rewards),
        final_criterion_value = criterion_value(candidate.protocol, hook.rewards),
        winner_run_id = string(stop_signal["winner_run_id"]),
        rewards = copy(hook.rewards),
        rewards_all_timesteps = copy(hook.rewards_all_timesteps),
        rewards_compare = copy(hook.rewards_compare),
        errored_episodes = copy(hook.errored_episodes),
        best_reward = hook.bestreward,
        best_episode = hook.bestepisode,
        continuation_trace = copy(continuation_trace),
        elapsed_seconds = Float64(elapsed_seconds),
        completed_at = string(now()),
        hostname = gethostname(),
        julia_version = string(VERSION),
        agent = agent,
    )
end

function save_failure!(candidate, results_directory, parent_path, parent_sha, message)
    path = joinpath(worker_directory(results_directory, candidate), "failure.jld2")
    atomic_jldsave(
        path;
        schema_version = SCHEMA_VERSION,
        status = "failed",
        protocol = candidate.protocol,
        rank = candidate.rank,
        run_id = candidate.run_id,
        parent_checkpoint_path = abspath(parent_path),
        parent_checkpoint_sha256 = parent_sha,
        failed_at = string(now()),
        error_message = message,
    )
end

function run_training_worker(;
    protocol,
    run_id,
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    source_results_directory::AbstractString = DEFAULT_SOURCE_RESULTS_DIRECTORY,
)
    candidate = candidate_for(protocol, run_id)
    freeze_selection_manifest!(; results_directory, source_results_directory)
    parent_path = source_checkpoint_path(source_results_directory, candidate)
    parent_sha = source_hash(parent_path)
    output_directory = worker_directory(results_directory, candidate)
    mkpath(output_directory)
    lock_path = joinpath(output_directory, ".worker.lock")
    lock = acquire_lock(lock_path)
    try
        if final_result_complete(final_path(results_directory, candidate), candidate, parent_sha)
            println("Complete result already exists; skipping $(candidate.protocol)/$(candidate.run_id).")
            return final_path(results_directory, candidate)
        end

        include_run_file!(candidate, joinpath(output_directory, "runtime"))
        state = load_training_state(candidate, results_directory, source_results_directory, parent_sha)
        global agent = state.agent
        restore_hook!(state)
        source_trace = read_source_metadata(parent_path).trace
        continuation_trace = copy(state.continuation_trace)
        configure_episode_initializer!(candidate, source_trace, continuation_trace)
        additional_episodes = state.additional_episodes
        elapsed_seconds = state.prior_elapsed_seconds

        hook(PRE_EXPERIMENT_STAGE, agent, env)
        agent(PRE_EXPERIMENT_STAGE, env)
        existing_signal = read_stop_signal(results_directory, candidate.protocol)
        if isnothing(existing_signal)
            println(
                "Starting $(candidate.protocol) rank $(candidate.rank) $(candidate.run_id) from " *
                "$(length(hook.rewards)) total episodes.",
            )
            flush(stdout)
        end

        while isnothing(existing_signal)
            episode_elapsed = @elapsed episode_reward = Base.invokelatest(train_one_episode!)
            elapsed_seconds += episode_elapsed
            additional_episodes += 1
            candidate.protocol === :varying && length(continuation_trace) != additional_episodes &&
                error("Continuation trace and additional-episode count differ.")
            save_resume!(
                candidate,
                results_directory,
                parent_path,
                parent_sha,
                additional_episodes,
                continuation_trace,
                elapsed_seconds,
            )

            metric = criterion_value(candidate.protocol, hook.rewards)
            reached = criterion_reached(candidate.protocol, hook.rewards)
            @printf(
                "[%s] protocol=%s rank=%d run=%s additional=%d total=%d episode_reward=%.6f criterion_value=%.6f target=%.6f reached=%s\n",
                now(),
                candidate.protocol,
                candidate.rank,
                candidate.run_id,
                additional_episodes,
                length(hook.rewards),
                episode_reward,
                metric,
                candidate.protocol === :fixed ? FIXED_THRESHOLD : VARYING_THRESHOLD,
                reached,
            )
            flush(stdout)

            if reached
                won = publish_candidate!(candidate, results_directory, metric, additional_episodes)
                won && println("Published protocol stop signal as winner $(candidate.run_id).")
            end
            existing_signal = read_stop_signal(results_directory, candidate.protocol)
        end

        hook(POST_EXPERIMENT_STAGE, agent, env)
        save_final!(
            candidate,
            results_directory,
            parent_path,
            parent_sha,
            additional_episodes,
            continuation_trace,
            elapsed_seconds,
            existing_signal,
        )
        winner = string(existing_signal["winner_run_id"]) == candidate.run_id
        println(
            "Saved final checkpoint for $(candidate.run_id); " *
            "winner=$winner, additional_episodes=$additional_episodes.",
        )
        flush(stdout)
        return final_path(results_directory, candidate)
    catch error_value
        message = sprint(showerror, error_value, catch_backtrace())
        save_failure!(candidate, results_directory, parent_path, parent_sha, message)
        rethrow(error_value)
    finally
        isdir(lock) && rm(lock; recursive = true, force = true)
    end
end

function candidate_ready(results_directory, protocol)
    protocol = normalize_protocol(protocol)
    manifest_path = candidate_manifest_path(results_directory, protocol)
    isfile(manifest_path) || return false
    manifest = JLD2.load(manifest_path)
    final_checkpoint = string(manifest["final_checkpoint_path"])
    isfile(final_checkpoint) || return false
    return try
        JLD2.jldopen(final_checkpoint, "r") do file
            string(read(file, "status")) == "complete" &&
            Bool(read(file, "is_protocol_winner")) &&
            string(read(file, "run_id")) == string(manifest["winner_run_id"])
        end
    catch
        false
    end
end

function protocol_ready_for_test(results_directory, protocol)
    protocol = normalize_protocol(protocol)
    candidate_ready(results_directory, protocol) || return false
    protocol_candidates = [candidate for candidate in CANDIDATES if candidate.protocol === protocol]
    return all(protocol_candidates) do candidate
        path = final_path(results_directory, candidate)
        isfile(path) || return false
        try
            JLD2.jldopen(path, "r") do file
                string(read(file, "status")) == "complete" &&
                string(read(file, "run_id")) == candidate.run_id &&
                Symbol(read(file, "protocol")) === protocol
            end
        catch
            false
        end
    end
end

test_directory(results_directory, protocol) = joinpath(
    results_directory,
    "test",
    string(normalize_protocol(protocol)),
)

test_complete(results_directory, protocol) = begin
    path = joinpath(test_directory(results_directory, protocol), "summary.jld2")
    isfile(path) || return false
    try
        JLD2.load(path, "status") == "complete"
    catch
        false
    end
end

expert_export_directory(results_directory, protocol) = joinpath(
    results_directory,
    "experts",
    string(normalize_protocol(protocol)),
)

expert_export_path(results_directory, protocol) = joinpath(
    expert_export_directory(results_directory, protocol),
    "expert.jld2",
)

distillation_expert_path(distillation_expert_directory, protocol) = joinpath(
    distillation_expert_directory,
    string(normalize_protocol(protocol)),
    "agent.jld2",
)

expert_publication_manifest_path(results_directory) = joinpath(
    results_directory,
    "experts",
    "publication_manifest.jld2",
)

function compact_agent_trajectory!(saved_agent)
    hasproperty(saved_agent, :trajectory) || error("The winner agent has no trajectory field.")
    trajectory = saved_agent.trajectory
    empty!(trajectory)
    for (trace_name, trace) in Base.pairs(trajectory.traces)
        hasproperty(trace, :buffer) || error(
            "Trajectory trace '$trace_name' has no shrinkable backing buffer.",
        )
        old_buffer = trace.buffer
        dimensions = ntuple(
            dimension -> dimension == ndims(old_buffer) ? 1 : size(old_buffer, dimension),
            ndims(old_buffer),
        )
        trace.buffer = similar(old_buffer, dimensions)
        trace.first = 1
        trace.nframes = 0
    end
    validate_compact_agent(saved_agent)
    return saved_agent
end

function validate_compact_agent(saved_agent)
    isempty(saved_agent.trajectory) || error("Compacted expert trajectory is not empty.")
    all(
        size(trace.buffer, ndims(trace.buffer)) == 1
        for trace in Base.values(saved_agent.trajectory.traces)
    ) || error("A compacted expert trajectory buffer still has capacity above one.")
    return true
end

function validate_agent_only_checkpoint(path)
    keys_in_file = JLD2.jldopen(path, "r") do file
        sort!(String.(collect(keys(file))))
    end
    keys_in_file == ["agent"] || error("Expert checkpoint contains unexpected entries: $path")
    saved_agent = JLD2.load(path, "agent")
    validate_compact_agent(saved_agent)
    return true
end

function experts_published(
    results_directory,
    distillation_expert_directory::AbstractString = DEFAULT_DISTILLATION_EXPERT_DIRECTORY,
)
    path = expert_publication_manifest_path(results_directory)
    isfile(path) || return false
    return try
        manifest = JLD2.load(path)
        string(manifest["status"]) == "complete" || return false
        records = manifest["experts"]
        all(records) do record
            export_path = string(record.export_path)
            target_path = string(record.distillation_path)
            expected_hash = string(record.expert_sha256)
            isfile(export_path) && isfile(target_path) &&
                source_hash(export_path) == expected_hash &&
                source_hash(target_path) == expected_hash
        end
    catch
        false
    end
end

function publish_distillation_experts!(;
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
    distillation_expert_directory::AbstractString = DEFAULT_DISTILLATION_EXPERT_DIRECTORY,
)
    all(protocol -> test_complete(results_directory, protocol), PROTOCOLS) || error(
        "Both winner test evaluations must be complete before expert publication.",
    )
    experts_published(results_directory, distillation_expert_directory) && begin
        println("Both compact experts are already published; skipping replacement.")
        return expert_publication_manifest_path(results_directory)
    end

    root = joinpath(results_directory, "experts")
    mkpath(root)
    return with_lock(joinpath(root, ".publication.lock"); wait_seconds = 60.0) do
        experts_published(results_directory, distillation_expert_directory) &&
            return expert_publication_manifest_path(results_directory)

        selected = map(PROTOCOLS) do protocol
            manifest = JLD2.load(candidate_manifest_path(results_directory, protocol))
            winner_run_id = string(manifest["winner_run_id"])
            checkpoint_path = string(manifest["final_checkpoint_path"])
            isfile(checkpoint_path) || error("Winner checkpoint is missing: $checkpoint_path")
            saved_agent = JLD2.load(checkpoint_path, "agent")
            compact_agent_trajectory!(saved_agent)
            (
                protocol,
                winner_run_id,
                checkpoint_path = abspath(checkpoint_path),
                checkpoint_sha256 = source_hash(checkpoint_path),
                agent = saved_agent,
                export_path = abspath(expert_export_path(results_directory, protocol)),
                distillation_path = abspath(
                    distillation_expert_path(distillation_expert_directory, protocol),
                ),
            )
        end

        staged = NamedTuple[]
        try
            for record in selected
                mkpath(dirname(record.export_path))
                mkpath(dirname(record.distillation_path))
                token = "$(getpid()).$(time_ns()).$(uuid4())"
                export_temporary = joinpath(
                    dirname(record.export_path),
                    ".$(basename(record.export_path)).$token.tmp",
                )
                distillation_temporary = joinpath(
                    dirname(record.distillation_path),
                    ".$(basename(record.distillation_path)).$token.tmp",
                )
                JLD2.jldsave(export_temporary; agent = record.agent)
                validate_agent_only_checkpoint(export_temporary)
                cp(export_temporary, distillation_temporary; force = true)
                validate_agent_only_checkpoint(distillation_temporary)
                push!(staged, merge(record, (; export_temporary, distillation_temporary)))
            end

            for record in staged
                mv(record.export_temporary, record.export_path; force = true)
            end
            for record in staged
                mv(record.distillation_temporary, record.distillation_path; force = true)
            end

            publication_records = map(staged) do record
                expert_sha = source_hash(record.export_path)
                source_hash(record.distillation_path) == expert_sha || error(
                    "Published Distillation expert differs from its compact export.",
                )
                (
                    protocol = record.protocol,
                    winner_run_id = record.winner_run_id,
                    source_checkpoint_path = record.checkpoint_path,
                    source_checkpoint_sha256 = record.checkpoint_sha256,
                    export_path = record.export_path,
                    distillation_path = record.distillation_path,
                    expert_sha256 = expert_sha,
                    bytes = filesize(record.export_path),
                )
            end
            manifest_path = expert_publication_manifest_path(results_directory)
            atomic_jldsave(
                manifest_path;
                schema_version = SCHEMA_VERSION,
                status = "complete",
                published_at = string(now()),
                experts = publication_records,
            )
            println("Published compact Fixed and Varying experts and replaced Distillation experts.")
            for record in publication_records
                println(
                    "  $(record.protocol): $(record.export_path) -> " *
                    "$(record.distillation_path) ($(record.bytes) bytes)",
                )
            end
            flush(stdout)
            return manifest_path
        finally
            for record in staged
                isfile(record.export_temporary) && rm(record.export_temporary; force = true)
                isfile(record.distillation_temporary) &&
                    rm(record.distillation_temporary; force = true)
            end
        end
    end
end

function test_cases(protocol)
    protocol === :fixed && return [(case_id = "fixed_shared", choice = nothing)]
    raw = JLD2.load(CORPUS_PATH, "corpus")
    split = haskey(raw, :test) ? raw[:test] : raw["test"]
    base_seeds = sort!(Int.(collect(keys(split))))
    length(base_seeds) == 2 || error("Expected two Varying test bases, got $(length(base_seeds)).")
    return [
        (
            case_id = "base_$(base_seed)_mirror_$(Int(mirror))_offset_$(offset)",
            choice = (split = :test, base_seed, mirror, offset),
        )
        for base_seed in base_seeds, mirror in (false, true), offset in (0, 20)
    ] |> vec
end

function reset_test_episode!(protocol, choice)
    initial_state = if protocol === :fixed
        Base.invokelatest(generate_random_init)
    else
        first(Base.invokelatest(
            generate_random_init;
            split = choice.split,
            base_seed = choice.base_seed,
            mirror = choice.mirror,
            offset = choice.offset,
        ))
    end
    env.y0 = initial_state
    Base.invokelatest(reset!, env)
    return nothing
end

function deterministic_test_rollout(protocol, choice)
    reset_test_episode!(protocol, choice)
    rewards = Float64[]
    while !(is_terminated(env) || is_truncated(env))
        action = RL.prob(agent.policy, env).μ
        hasproperty(agent.policy, :clip1) && agent.policy.clip1 && clamp!(action, -1.0, 1.0)
        env(action)
        push!(rewards, mean(Float64.(reward(env))))
    end
    length(rewards) == TEST_EPISODE_STEPS || error(
        "Expected $TEST_EPISODE_STEPS test steps, got $(length(rewards)).",
    )
    return rewards
end

function write_test_csv(path, records)
    open(path, "w") do io
        println(io, "case_id,base_seed,mirror,offset,score")
        for record in records
            choice = record.choice
            println(io, join((
                record.case_id,
                isnothing(choice) ? "" : choice.base_seed,
                isnothing(choice) ? "" : choice.mirror,
                isnothing(choice) ? "" : choice.offset,
                record.score,
            ), ','))
        end
    end
    return path
end

function plot_test_rewards(protocol, records, output_directory)
    steps = collect(1:TEST_EPISODE_STEPS)
    traces = PlotlyJS.GenericTrace[]
    if protocol === :fixed
        push!(traces, scatter(
            x = steps,
            y = only(records).rewards,
            mode = "lines",
            name = "Fixed test episode",
            line = attr(color = "#2166AC", width = 3),
        ))
    else
        for record in records
            push!(traces, scatter(
                x = steps,
                y = record.rewards,
                mode = "lines",
                name = record.case_id,
                line = attr(width = 1),
                opacity = 0.35,
                showlegend = false,
            ))
        end
        mean_curve = vec(mean(reduce(hcat, getproperty.(records, :rewards)); dims = 2))
        push!(traces, scatter(
            x = steps,
            y = mean_curve,
            mode = "lines",
            name = "Mean over 8 test cases",
            line = attr(color = "#B2182B", width = 3),
        ))
    end
    plot_handle = Plot(
        traces,
        Layout(
            template = "plotly_white",
            title = attr(
                text = "$(uppercasefirst(string(protocol)))-IC expert candidate — test reward curves",
                x = 0.5,
                xanchor = "center",
            ),
            paper_bgcolor = "white",
            plot_bgcolor = "white",
            width = 1000,
            height = 600,
            margin = attr(l = 95, r = 35, t = 80, b = 80),
            xaxis = attr(title = "Control step", gridcolor = "#E6E6E6"),
            yaxis = attr(
                title = "Mean environment reward (higher is better)",
                gridcolor = "#E6E6E6",
            ),
            hovermode = "x unified",
        ),
    )
    svg_path = joinpath(output_directory, "reward_curves.svg")
    png_path = joinpath(output_directory, "reward_curves.png")
    PlotlyJS.savefig(plot_handle, svg_path; width = 1000, height = 600)
    PlotlyJS.savefig(plot_handle, png_path; width = 1000, height = 600)
    return (; svg_path, png_path)
end

function run_test_protocol_worker(;
    protocol,
    results_directory::AbstractString = DEFAULT_RESULTS_DIRECTORY,
)
    protocol = normalize_protocol(protocol)
    protocol_ready_for_test(results_directory, protocol) || error(
        "Winner or one of the ten final protocol checkpoints for $protocol is not ready.",
    )
    test_complete(results_directory, protocol) && begin
        println("Complete test result already exists for $protocol; skipping.")
        return joinpath(test_directory(results_directory, protocol), "summary.jld2")
    end

    manifest = JLD2.load(candidate_manifest_path(results_directory, protocol))
    winner_run_id = string(manifest["winner_run_id"])
    candidate = candidate_for(protocol, winner_run_id)
    final_checkpoint = string(manifest["final_checkpoint_path"])
    final_sha = source_hash(final_checkpoint)
    output_directory = test_directory(results_directory, protocol)
    mkpath(output_directory)
    lock = acquire_lock(joinpath(output_directory, ".worker.lock"))
    try
        include_run_file!(candidate, joinpath(output_directory, "runtime"))
        global agent = JLD2.load(final_checkpoint, "agent")
        Flux.testmode!(agent.policy)
        records = NamedTuple[]
        for case in test_cases(protocol)
            cache_path = joinpath(output_directory, "episodes", "$(case.case_id).jld2")
            record = if isfile(cache_path)
                cached = JLD2.load(cache_path)
                string(cached["status"]) == "complete" || error("Incomplete test cache: $cache_path")
                string(cached["checkpoint_sha256"]) == final_sha ||
                    error("Stale test cache for a different candidate: $cache_path")
                (
                    case_id = case.case_id,
                    choice = case.choice,
                    rewards = Float64.(cached["rewards"]),
                    score = Float64(cached["score"]),
                )
            else
                rewards = deterministic_test_rollout(protocol, case.choice)
                score = sum(rewards)
                atomic_jldsave(
                    cache_path;
                    schema_version = SCHEMA_VERSION,
                    status = "complete",
                    protocol,
                    winner_run_id,
                    checkpoint_path = abspath(final_checkpoint),
                    checkpoint_sha256 = final_sha,
                    case_id = case.case_id,
                    choice = case.choice,
                    policy = "deterministic_mean_action",
                    rewards,
                    score,
                    completed_at = string(now()),
                )
                println("Completed $protocol test case $(case.case_id): score=$score")
                flush(stdout)
                (; case_id = case.case_id, choice = case.choice, rewards, score)
            end
            push!(records, record)
        end

        plots = plot_test_rewards(protocol, records, output_directory)
        csv_path = write_test_csv(joinpath(output_directory, "scores.csv"), records)
        scores = getproperty.(records, :score)
        summary_path = joinpath(output_directory, "summary.jld2")
        atomic_jldsave(
            summary_path;
            schema_version = SCHEMA_VERSION,
            status = "complete",
            protocol,
            winner_run_id,
            checkpoint_path = abspath(final_checkpoint),
            checkpoint_sha256 = final_sha,
            policy = "deterministic_mean_action",
            cases = records,
            scores,
            mean_score = mean(scores),
            csv_path = abspath(csv_path),
            reward_plot_svg = abspath(plots.svg_path),
            reward_plot_png = abspath(plots.png_path),
            completed_at = string(now()),
        )
        println(
            "Completed $protocol candidate test: mean_score=$(mean(scores)); " *
            "plot=$(plots.svg_path)",
        )
        flush(stdout)
        return summary_path
    finally
        isdir(lock) && rm(lock; recursive = true, force = true)
    end
end

end
