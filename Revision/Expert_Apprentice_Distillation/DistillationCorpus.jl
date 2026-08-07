using Dates
using JLD2
using SHA

const DISTILLATION_SCHEMA_VERSION = 1
const DISTILLATION_PROTOCOLS = (:fixed, :varying)
const DISTILLATION_SPLITS = (:train, :validation, :test)
const DISTILLATION_SHARED_SPLIT = :shared
const DISTILLATION_ROLLOUT_STEPS = 200
const DISTILLATION_TRAIN_OFFSETS = 0:95
const DISTILLATION_EVALUATION_OFFSETS = (0, 20)

const DISTILLATION_CHANNELS = 3
const DISTILLATION_HORIZONTAL_SENSORS = 48
const DISTILLATION_VERTICAL_SENSORS = 8
const DISTILLATION_ACTUATORS = 12
const DISTILLATION_WINDOW_SIZE = 15
const DISTILLATION_EXPECTED_BASES = Dict(:train => 20, :validation => 1, :test => 2)

const DISTILLATION_WORKER_DIRECTORY = get(
    ENV,
    "DISTILLATION_WORKER_DIRECTORY",
    joinpath(@__DIR__, "worker_results"),
)

function normalize_distillation_protocol(protocol)::Symbol
    value = Symbol(lowercase(string(protocol)))
    value = get(Dict(:fic => :fixed, :ric => :varying), value, value)
    value in DISTILLATION_PROTOCOLS || throw(
        ArgumentError("Unknown protocol '$protocol'. Use :fixed or :varying."),
    )
    return value
end

function normalize_distillation_split(split; allow_shared::Bool = false)::Symbol
    value = Symbol(lowercase(string(split)))
    value = get(
        Dict(:training => :train, :val => :validation, :valid => :validation, :testing => :test),
        value,
        value,
    )
    allowed = allow_shared ? (DISTILLATION_SPLITS..., DISTILLATION_SHARED_SPLIT) : DISTILLATION_SPLITS
    value in allowed || throw(
        ArgumentError("Unknown split '$split'. Use :train, :validation, or :test."),
    )
    return value
end

function distillation_offsets(protocol, split = :train)
    protocol_value = normalize_distillation_protocol(protocol)
    protocol_value === :fixed && return [nothing]
    split_value = normalize_distillation_split(split)
    return split_value === :train ?
        collect(DISTILLATION_TRAIN_OFFSETS) : collect(DISTILLATION_EVALUATION_OFFSETS)
end

function distillation_value(container, key::Symbol)
    if haskey(container, key)
        return container[key]
    elseif haskey(container, string(key))
        return container[string(key)]
    end
    error("Distillation data has no '$key' entry.")
end

function empty_distillation_dataset()
    return Dict{Symbol, Any}(
        :observations => zeros(
            Float32,
            DISTILLATION_CHANNELS,
            DISTILLATION_HORIZONTAL_SENSORS,
            DISTILLATION_VERTICAL_SENSORS,
            0,
        ),
        :expert_actions => zeros(Float32, 1, DISTILLATION_ACTUATORS, 0),
        :episodes => Dict{Symbol, Any}[],
        :source_files => String[],
        :worker_count => 0,
        :sample_count => 0,
        :expert_identifier => nothing,
        :observation_metadata => nothing,
        :complete => false,
        :coverage_complete => false,
    )
end

function empty_distillation_corpus()
    fixed_shared = empty_distillation_dataset()
    return Dict{Symbol, Any}(
        :fixed => Dict{Symbol, Any}(
            :train => fixed_shared,
            :validation => fixed_shared,
            :test => fixed_shared,
        ),
        :varying => Dict{Symbol, Any}(
            split => empty_distillation_dataset() for split in DISTILLATION_SPLITS
        ),
    )
end

function distillation_worker_path(
    protocol,
    split = DISTILLATION_SHARED_SPLIT;
    base_seed::Union{Nothing, Integer} = nothing,
    mirror::Bool = false,
    worker_directory::AbstractString = DISTILLATION_WORKER_DIRECTORY,
)
    protocol_value = normalize_distillation_protocol(protocol)
    if protocol_value === :fixed
        return joinpath(worker_directory, "fixed", "fixed_shared.jld2")
    end

    split_value = normalize_distillation_split(split)
    isnothing(base_seed) && throw(ArgumentError("base_seed is required for varying IC."))
    filename = "base_$(Int(base_seed))_mirror_$(mirror ? 1 : 0).jld2"
    return joinpath(worker_directory, "varying", string(split_value), filename)
end

function is_distillation_worker_file(path::AbstractString)
    return endswith(lowercase(path), ".jld2") &&
           !occursin(".tmp.", basename(path)) &&
           !startswith(basename(path), ".")
end

function available_distillation_worker_files(
    worker_directory::AbstractString = DISTILLATION_WORKER_DIRECTORY,
)
    isdir(worker_directory) || return String[]
    paths = String[]
    for (directory, _, filenames) in walkdir(worker_directory)
        for filename in filenames
            path = joinpath(directory, filename)
            is_distillation_worker_file(path) && push!(paths, path)
        end
    end
    sort!(paths)
    return paths
end

function atomic_save_distillation_worker(path::AbstractString, worker_result)
    mkpath(dirname(path))
    temporary_path = path * ".tmp.$(getpid()).$(time_ns())"
    try
        JLD2.jldsave(
            temporary_path;
            worker_result,
            schema_version = DISTILLATION_SCHEMA_VERSION,
        )
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return path
end

function normalize_expert_actions(action)
    values = Float32.(Array(action))
    if ndims(values) == 1
        values = reshape(values, 1, :)
    elseif ndims(values) == 3 && size(values, 3) == 1
        values = dropdims(values; dims = 3)
    end
    size(values) == (1, DISTILLATION_ACTUATORS) || error(
        "Expected expert action means of size (1, $(DISTILLATION_ACTUATORS)), got $(size(values)).",
    )
    return values
end

"""
    global_sensor_observation(fields; horizontal_indices, vertical_indices,
                              add_joon_position_encoding=true)

Extract the unique global `3×48×8` sensor tensor used to reconstruct the
overlapping MAT observation windows. The optional sinusoidal encoding matches
the current revision MAT run files.
"""
function global_sensor_observation(
    fields;
    horizontal_indices,
    vertical_indices,
    add_joon_position_encoding::Bool = true,
)
    observation = Array(fields[:, horizontal_indices, vertical_indices])
    expected_size = (
        DISTILLATION_CHANNELS,
        DISTILLATION_HORIZONTAL_SENSORS,
        DISTILLATION_VERTICAL_SENSORS,
    )
    size(observation) == expected_size || error(
        "Expected global sensor tensor size $expected_size, got $(size(observation)).",
    )

    if add_joon_position_encoding
        for horizontal_index in 1:DISTILLATION_HORIZONTAL_SENSORS
            phase = (2 * pi / DISTILLATION_HORIZONTAL_SENSORS) * horizontal_index
            observation[1, horizontal_index, :] .+= sin(phase)
        end
    end
    return Float32.(observation)
end

"""
    local_mat_observation(global_observation; actuator_sensor_indices,
                          window_size=15)

Reconstruct the exact overlapping `360×12` MAT observation from a global
sensor tensor without storing duplicate local input rows in the corpus.
"""
function local_mat_observation(
    global_observation;
    actuator_sensor_indices,
    window_size::Integer = DISTILLATION_WINDOW_SIZE,
)
    size(global_observation) == (
        DISTILLATION_CHANNELS,
        DISTILLATION_HORIZONTAL_SENSORS,
        DISTILLATION_VERTICAL_SENSORS,
    ) || error("Unexpected global observation size $(size(global_observation)).")
    length(actuator_sensor_indices) == DISTILLATION_ACTUATORS || error(
        "Expected $(DISTILLATION_ACTUATORS) actuator sensor indices.",
    )
    isodd(window_size) || throw(ArgumentError("window_size must be odd."))

    half_width = fld(window_size, 2)
    local_rows = DISTILLATION_CHANNELS * window_size * DISTILLATION_VERTICAL_SENSORS
    result = Matrix{Float32}(undef, local_rows, DISTILLATION_ACTUATORS)
    for (actor_index, sensor_index) in enumerate(actuator_sensor_indices)
        horizontal_indices = [
            mod1(Int(sensor_index) + offset, DISTILLATION_HORIZONTAL_SENSORS)
            for offset in -half_width:half_width
        ]
        result[:, actor_index] .= vec(global_observation[:, horizontal_indices, :])
    end
    return result
end

"""
    local_mat_observation_batch(global_observations; actuator_sensor_indices,
                                window_size=15)

Reconstruct only the requested in-memory mini-batch as `360×12×N`. This keeps
the compact corpus representation intact instead of materializing all
overlapping windows for the complete training set.
"""
function local_mat_observation_batch(
    global_observations;
    actuator_sensor_indices,
    window_size::Integer = DISTILLATION_WINDOW_SIZE,
)
    ndims(global_observations) == 4 || error(
        "Expected a four-dimensional global observation batch, got $(size(global_observations)).",
    )
    size(global_observations)[1:3] == (
        DISTILLATION_CHANNELS,
        DISTILLATION_HORIZONTAL_SENSORS,
        DISTILLATION_VERTICAL_SENSORS,
    ) || error("Unexpected global observation batch size $(size(global_observations)).")
    local_rows = DISTILLATION_CHANNELS * window_size * DISTILLATION_VERTICAL_SENSORS
    batch_size = size(global_observations, 4)
    result = Array{Float32}(undef, local_rows, DISTILLATION_ACTUATORS, batch_size)
    for sample_index in 1:batch_size
        result[:, :, sample_index] .= local_mat_observation(
            view(global_observations, :, :, :, sample_index);
            actuator_sensor_indices,
            window_size,
        )
    end
    return result
end

function distillation_batch(
    dataset,
    sample_indices;
    actuator_sensor_indices,
    window_size::Integer = DISTILLATION_WINDOW_SIZE,
)
    indices = Int.(collect(sample_indices))
    sample_count = Int(dataset[:sample_count])
    all(index -> 1 <= index <= sample_count, indices) || throw(
        BoundsError(1:sample_count, indices),
    )
    global_batch = view(dataset[:observations], :, :, :, indices)
    return (
        observations = local_mat_observation_batch(
            global_batch;
            actuator_sensor_indices,
            window_size,
        ),
        expert_actions = Array(view(dataset[:expert_actions], :, :, indices)),
        sample_indices = indices,
    )
end

function assert_lossless_observation_reconstruction(
    global_observation,
    expected_local_observation;
    actuator_sensor_indices,
    window_size::Integer = DISTILLATION_WINDOW_SIZE,
)
    reconstructed = local_mat_observation(
        global_observation;
        actuator_sensor_indices,
        window_size,
    )
    expected = Float32.(Array(expected_local_observation))
    if reconstructed != expected
        maximum_error = maximum(abs.(reconstructed .- expected))
        mismatch_count = count(value -> !iszero(value), reconstructed .- expected)
        error(
            "Global sensor storage does not reconstruct the current MAT observation exactly " *
            "($mismatch_count differing values, maximum absolute error $maximum_error).",
        )
    end
    return true
end

function normalize_episode_specs(protocol, split, base_seed, mirror, offsets)
    protocol_value = normalize_distillation_protocol(protocol)
    if protocol_value === :fixed
        return [
            (
                split = DISTILLATION_SHARED_SPLIT,
                base_seed = nothing,
                mirror = false,
                offset = nothing,
            ),
        ]
    end

    split_value = normalize_distillation_split(split)
    isnothing(base_seed) && throw(ArgumentError("base_seed is required for varying IC."))
    requested_offsets = isnothing(offsets) ?
        distillation_offsets(protocol_value, split_value) : collect(offsets)
    normalized_offsets = sort!(unique(mod.(Int.(requested_offsets), 96)))
    isempty(normalized_offsets) && throw(ArgumentError("At least one offset is required."))
    return [
        (
            split = split_value,
            base_seed = Int(base_seed),
            mirror = Bool(mirror),
            offset = offset,
        )
        for offset in normalized_offsets
    ]
end

"""
    generate_distillation_worker!(; ...)

Generate all episodes owned by one worker. Varying-IC workers own one
`(split, base_seed, mirror)` combination. Training workers evaluate all 96
offsets; validation and test workers evaluate the fixed offsets 0 and 20.
The Fixed-IC worker owns one shared episode. Environment-specific behavior is
supplied through four callbacks, so every Package-6/7/8 worker uses this same
generation function.
"""
function generate_distillation_worker!(
    ;
    protocol,
    split = DISTILLATION_SHARED_SPLIT,
    base_seed::Union{Nothing, Integer} = nothing,
    mirror::Bool = false,
    offsets = nothing,
    rollout_steps::Integer = DISTILLATION_ROLLOUT_STEPS,
    initialize_episode!::Function,
    observe::Function,
    expert_mean::Function,
    advance!::Function,
    expert_metadata = Dict{Symbol, Any}(),
    observation_metadata = Dict{Symbol, Any}(),
    run_seed::Integer = 0,
    worker_directory::AbstractString = DISTILLATION_WORKER_DIRECTORY,
    overwrite::Bool = false,
)
    rollout_steps > 0 || throw(ArgumentError("rollout_steps must be positive."))
    protocol_value = normalize_distillation_protocol(protocol)
    episode_specs = normalize_episode_specs(protocol_value, split, base_seed, mirror, offsets)
    output_path = distillation_worker_path(
        protocol_value,
        protocol_value === :fixed ? DISTILLATION_SHARED_SPLIT : split;
        base_seed,
        mirror,
        worker_directory,
    )

    if isfile(output_path) && !overwrite
        loaded = load_distillation_worker(output_path)
        distillation_value(loaded, :complete) === true || error(
            "Existing worker file is incomplete: $output_path",
        )
        requested_offsets = [spec.offset for spec in episode_specs]
        loaded_identifier = string(get(
            distillation_value(loaded, :expert_metadata),
            :identifier,
            get(distillation_value(loaded, :expert_metadata), "identifier", "unknown"),
        ))
        requested_identifier = string(get(
            expert_metadata,
            :identifier,
            get(expert_metadata, "identifier", "unknown"),
        ))
        matching = Int(distillation_value(loaded, :rollout_steps)) == rollout_steps &&
                   collect(distillation_value(loaded, :offsets)) == requested_offsets &&
                   loaded_identifier == requested_identifier
        matching || error(
            "Existing worker file does not match the requested rollout/expert: $output_path. " *
            "Pass overwrite=true only if replacement is intended.",
        )
        return output_path
    end

    lock_path = output_path * ".lock"
    mkpath(dirname(output_path))
    try
        mkdir(lock_path)
    catch error_value
        isdir(lock_path) && error("Another worker owns lock '$lock_path'.")
        rethrow(error_value)
    end

    try
        sample_count = length(episode_specs) * Int(rollout_steps)
        observations = Array{Float32}(undef,
            DISTILLATION_CHANNELS,
            DISTILLATION_HORIZONTAL_SENSORS,
            DISTILLATION_VERTICAL_SENSORS,
            sample_count,
        )
        expert_actions = Array{Float32}(undef, 1, DISTILLATION_ACTUATORS, sample_count)
        episodes = Dict{Symbol, Any}[]

        sample_index = 1
        for (episode_index, spec) in enumerate(episode_specs)
            initialize_episode!(spec)
            sample_start = sample_index
            for control_step in 1:rollout_steps
                observation = Float32.(Array(observe()))
                expected_observation_size = (
                    DISTILLATION_CHANNELS,
                    DISTILLATION_HORIZONTAL_SENSORS,
                    DISTILLATION_VERTICAL_SENSORS,
                )
                size(observation) == expected_observation_size || error(
                    "Observation callback returned $(size(observation)); expected $expected_observation_size.",
                )
                action = normalize_expert_actions(expert_mean())

                observations[:, :, :, sample_index] .= observation
                expert_actions[:, :, sample_index] .= action
                advance!(action)
                sample_index += 1
            end
            push!(episodes, Dict{Symbol, Any}(
                :episode_index => episode_index,
                :split => spec.split,
                :base_seed => spec.base_seed,
                :mirror => spec.mirror,
                :offset => spec.offset,
                :sample_start => sample_start,
                :sample_stop => sample_index - 1,
                :rollout_steps => Int(rollout_steps),
            ))
        end

        result = Dict{Symbol, Any}(
            :schema_version => DISTILLATION_SCHEMA_VERSION,
            :complete => true,
            :protocol => protocol_value,
            :split => protocol_value === :fixed ? DISTILLATION_SHARED_SPLIT : normalize_distillation_split(split),
            :base_seed => protocol_value === :fixed ? nothing : Int(base_seed),
            :mirror => protocol_value === :fixed ? false : mirror,
            :offsets => [spec.offset for spec in episode_specs],
            :rollout_steps => Int(rollout_steps),
            :run_seed => Int(run_seed),
            :observations => observations,
            :expert_actions => expert_actions,
            :episodes => episodes,
            :expert_metadata => Dict{Symbol, Any}(expert_metadata),
            :observation_metadata => Dict{Symbol, Any}(observation_metadata),
            :created_at => string(Dates.now()),
            :storage_layout => "global_sensor_tensor_3x48x8",
        )
        atomic_save_distillation_worker(output_path, result)
        return output_path
    finally
        isdir(lock_path) && rm(lock_path; recursive = true, force = true)
    end
end

function load_distillation_worker(path::AbstractString)
    isfile(path) || error("Distillation worker file does not exist: $path")
    loaded = JLD2.load(path)
    haskey(loaded, "worker_result") || error("Worker file '$path' has no worker_result entry.")
    result = loaded["worker_result"]
    Int(distillation_value(result, :schema_version)) == DISTILLATION_SCHEMA_VERSION || error(
        "Unsupported distillation schema in '$path'.",
    )
    distillation_value(result, :complete) === true || error("Incomplete worker file '$path'.")
    return result
end

function worker_identity(result)
    protocol = normalize_distillation_protocol(distillation_value(result, :protocol))
    if protocol === :fixed
        return (:fixed, DISTILLATION_SHARED_SPLIT, nothing, false)
    end
    return (
        :varying,
        normalize_distillation_split(distillation_value(result, :split)),
        Int(distillation_value(result, :base_seed)),
        Bool(distillation_value(result, :mirror)),
    )
end

function merge_distillation_workers(paths::Vector{String})
    isempty(paths) && return empty_distillation_dataset()
    identities = Set{Any}()
    total_samples = 0
    expert_identifiers = Set{String}()
    observation_metadata_reference = nothing

    for path in paths
        result = load_distillation_worker(path)
        identity = worker_identity(result)
        identity in identities && error("Duplicate distillation worker identity $identity.")
        push!(identities, identity)
        observations = distillation_value(result, :observations)
        actions = distillation_value(result, :expert_actions)
        size(observations)[1:3] == (
            DISTILLATION_CHANNELS,
            DISTILLATION_HORIZONTAL_SENSORS,
            DISTILLATION_VERTICAL_SENSORS,
        ) || error("Unexpected observation shape in '$path'.")
        size(actions)[1:2] == (1, DISTILLATION_ACTUATORS) || error(
            "Unexpected expert-action shape in '$path'.",
        )
        size(observations, 4) == size(actions, 3) || error(
            "Observation/action sample mismatch in '$path'.",
        )
        total_samples += size(observations, 4)
        expert_metadata = distillation_value(result, :expert_metadata)
        identifier = string(get(expert_metadata, :identifier, get(expert_metadata, "identifier", "unknown")))
        push!(expert_identifiers, identifier)
        observation_metadata = distillation_value(result, :observation_metadata)
        if isnothing(observation_metadata_reference)
            observation_metadata_reference = observation_metadata
        elseif observation_metadata != observation_metadata_reference
            error("Worker files use different observation configurations; mismatch at '$path'.")
        end
    end
    length(expert_identifiers) == 1 || error(
        "Worker files use different experts: $(sort!(collect(expert_identifiers))).",
    )

    observations = Array{Float32}(undef,
        DISTILLATION_CHANNELS,
        DISTILLATION_HORIZONTAL_SENSORS,
        DISTILLATION_VERTICAL_SENSORS,
        total_samples,
    )
    expert_actions = Array{Float32}(undef, 1, DISTILLATION_ACTUATORS, total_samples)
    episodes = Dict{Symbol, Any}[]
    destination_start = 1

    for path in paths
        result = load_distillation_worker(path)
        worker_observations = distillation_value(result, :observations)
        worker_actions = distillation_value(result, :expert_actions)
        worker_count = size(worker_observations, 4)
        destination_stop = destination_start + worker_count - 1
        observations[:, :, :, destination_start:destination_stop] .= worker_observations
        expert_actions[:, :, destination_start:destination_stop] .= worker_actions
        for raw_episode in distillation_value(result, :episodes)
            episode = Dict{Symbol, Any}(
                (key isa Symbol ? key : Symbol(key)) => value
                for (key, value) in raw_episode
            )
            episode[:sample_start] = Int(episode[:sample_start]) + destination_start - 1
            episode[:sample_stop] = Int(episode[:sample_stop]) + destination_start - 1
            episode[:source_file] = path
            push!(episodes, episode)
        end
        destination_start = destination_stop + 1
    end

    return Dict{Symbol, Any}(
        :observations => observations,
        :expert_actions => expert_actions,
        :episodes => episodes,
        :source_files => copy(paths),
        :worker_count => length(paths),
        :sample_count => total_samples,
        :expert_identifier => only(expert_identifiers),
        :observation_metadata => observation_metadata_reference,
        :complete => true,
    )
end

"""
    load_distillation_corpus(worker_directory=DISTILLATION_WORKER_DIRECTORY;
                               protocols=DISTILLATION_PROTOCOLS)

Load every complete worker JLD2 for the selected protocols and merge it in
memory by protocol and split.
Fixed IC uses one shared dataset object for train, validation, and test.
"""
function load_distillation_corpus(
    worker_directory::AbstractString = DISTILLATION_WORKER_DIRECTORY,
    ;
    protocols = DISTILLATION_PROTOCOLS,
)
    selected_protocols = Set(normalize_distillation_protocol.(collect(protocols)))
    isempty(selected_protocols) && throw(ArgumentError("protocols must not be empty."))
    corpus = empty_distillation_corpus()
    files = available_distillation_worker_files(worker_directory)
    fixed_files = String[]
    varying_files = Dict(split => String[] for split in DISTILLATION_SPLITS)

    fixed_directory = abspath(joinpath(worker_directory, "fixed"))
    varying_directory = abspath(joinpath(worker_directory, "varying"))
    path_is_below(path, directory) = begin
        relative = relpath(abspath(path), directory)
        relative != ".." && !startswith(relative, "..$(Base.Filesystem.path_separator)")
    end

    for path in files
        if path_is_below(path, fixed_directory) && !(:fixed in selected_protocols)
            continue
        elseif path_is_below(path, varying_directory) && !(:varying in selected_protocols)
            continue
        end
        result = load_distillation_worker(path)
        protocol = normalize_distillation_protocol(distillation_value(result, :protocol))
        protocol in selected_protocols || continue
        if protocol === :fixed
            push!(fixed_files, path)
        else
            split = normalize_distillation_split(distillation_value(result, :split))
            push!(varying_files[split], path)
        end
    end

    if !isempty(fixed_files)
        length(fixed_files) == 1 || error("Expected one Fixed-IC worker file, found $(length(fixed_files)).")
        fixed_dataset = merge_distillation_workers(fixed_files)
        fixed_dataset[:coverage_complete] = (
            fixed_dataset[:worker_count] == 1 &&
            length(fixed_dataset[:episodes]) == 1 &&
            fixed_dataset[:sample_count] == DISTILLATION_ROLLOUT_STEPS
        )
        for split in DISTILLATION_SPLITS
            corpus[:fixed][split] = fixed_dataset
        end
    end
    for split in DISTILLATION_SPLITS
        corpus[:varying][split] = merge_distillation_workers(varying_files[split])
        expected = expected_distillation_counts(:varying, split)
        dataset = corpus[:varying][split]
        dataset[:coverage_complete] = (
            dataset[:worker_count] == expected.workers &&
            length(dataset[:episodes]) == expected.episodes &&
            dataset[:sample_count] == expected.samples
        )
    end
    return corpus
end

function reload_distillation_corpus!(
    worker_directory::AbstractString = DISTILLATION_WORKER_DIRECTORY,
    ;
    protocols = DISTILLATION_PROTOCOLS,
)
    loaded = load_distillation_corpus(worker_directory; protocols)
    empty!(DISTILLATION_CORPUS)
    merge!(DISTILLATION_CORPUS, loaded)
    return DISTILLATION_CORPUS
end

function distillation_dataset(
    protocol,
    split = :train;
    corpus = DISTILLATION_CORPUS,
)
    protocol_value = normalize_distillation_protocol(protocol)
    split_value = normalize_distillation_split(split)
    return corpus[protocol_value][split_value]
end

function expected_distillation_counts(protocol, split = :train)
    protocol_value = normalize_distillation_protocol(protocol)
    split_value = normalize_distillation_split(split)
    if protocol_value === :fixed
        return (workers = 1, episodes = 1, samples = DISTILLATION_ROLLOUT_STEPS)
    end
    workers = DISTILLATION_EXPECTED_BASES[split_value] * 2
    episodes = workers * length(distillation_offsets(protocol_value, split_value))
    return (
        workers,
        episodes,
        samples = episodes * DISTILLATION_ROLLOUT_STEPS,
    )
end

function distillation_coverage(protocol, split = :train; corpus = DISTILLATION_CORPUS)
    dataset = distillation_dataset(protocol, split; corpus)
    expected = expected_distillation_counts(protocol, split)
    actual = (
        workers = Int(dataset[:worker_count]),
        episodes = length(dataset[:episodes]),
        samples = Int(dataset[:sample_count]),
    )
    return (
        expected,
        actual,
        complete = actual == expected,
    )
end

function assert_distillation_coverage(protocol, split = :train; corpus = DISTILLATION_CORPUS)
    coverage = distillation_coverage(protocol, split; corpus)
    coverage.complete || error(
        "Incomplete $(normalize_distillation_protocol(protocol))/$split distillation corpus: " *
        "expected $(coverage.expected), found $(coverage.actual).",
    )
    return true
end

function file_sha256(path::AbstractString)
    return open(path, "r") do io
        bytes2hex(SHA.sha256(io))
    end
end

function distillation_expert_candidates(protocol)
    protocol_value = normalize_distillation_protocol(protocol)
    env_name = protocol_value === :fixed ?
        "DISTILLATION_FIXED_EXPERT_PATH" : "DISTILLATION_VARYING_EXPERT_PATH"
    candidates = String[]
    haskey(ENV, env_name) && push!(candidates, abspath(ENV[env_name]))
    push!(
        candidates,
        joinpath(@__DIR__, "experts", string(protocol_value), "agent.jld2"),
    )
    if protocol_value === :fixed
        push!(candidates, joinpath(@__DIR__, "..", "Run_Files", "outputs", "Revision_FixedIC_MAT", "saves", "agent.jld2"))
    else
        push!(candidates, joinpath(@__DIR__, "..", "Run_Files", "outputs", "Revision_VaryingIC_MAT", "saves", "agentMAT.jld2"))
    end
    return unique(normpath.(candidates))
end

function find_distillation_expert(protocol; explicit_path = nothing)
    if !isnothing(explicit_path)
        path = abspath(string(explicit_path))
        isfile(path) || error("Requested expert checkpoint does not exist: $path")
        return path
    end
    for path in distillation_expert_candidates(protocol)
        isfile(path) && return path
    end
    return nothing
end

"""
    load_distillation_expert!(protocol; explicit_path=nothing,
                              allow_fresh_expert=false)

Replace the globally initialized MAT `agent` with a persisted expert when one
is available. A fresh MAT is accepted only through the explicit smoke-test
flag and is never a silent production fallback.
"""
function load_distillation_expert!(
    protocol;
    explicit_path = nothing,
    allow_fresh_expert::Bool = false,
)
    path = find_distillation_expert(protocol; explicit_path)
    if isnothing(path)
        allow_fresh_expert || error(
            "No $(normalize_distillation_protocol(protocol)) expert checkpoint found. " *
            "Set the protocol-specific DISTILLATION_*_EXPERT_PATH or pass an explicit path. " *
            "Use allow_fresh_expert=true only for technical smoke tests.",
        )
        isdefined(@__MODULE__, :agent) || error("No freshly initialized MAT agent is available.")
        run_seed = get(ENV, "REVISION_RUN_SEED", "unknown")
        return Dict{Symbol, Any}(
            :identifier => "fresh-initialized-mat-seed-$run_seed",
            :source => "fresh_initialized_smoke_test_only",
            :checkpoint_path => nothing,
            :checkpoint_sha256 => nothing,
        )
    end

    loaded = JLD2.load(path)
    expert = if haskey(loaded, "agent")
        loaded["agent"]
    elseif haskey(loaded, "expert")
        loaded["expert"]
    else
        error("Expert checkpoint '$path' has neither an 'agent' nor an 'expert' entry.")
    end
    global agent = expert
    digest = file_sha256(path)
    return Dict{Symbol, Any}(
        :identifier => "sha256:$digest",
        :source => "checkpoint",
        :checkpoint_path => path,
        :checkpoint_sha256 => digest,
    )
end


const DISTILLATION_CORPUS = if get(ENV, "DISTILLATION_SKIP_AUTOLOAD", "false") in ("1", "true", "yes")
    empty_distillation_corpus()
else
    requested_protocol = lowercase(get(ENV, "DISTILLATION_AUTOLOAD_PROTOCOL", "all"))
    requested_protocol in ("all", "fixed", "varying") || error(
        "DISTILLATION_AUTOLOAD_PROTOCOL must be all, fixed, or varying.",
    )
    selected_protocols = requested_protocol == "all" ?
        DISTILLATION_PROTOCOLS : (Symbol(requested_protocol),)
    load_distillation_corpus(; protocols = selected_protocols)
end
