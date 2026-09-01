module NoiseStudy

using Dates
using JLD2
using Printf
using Random
using SHA
using StableRNGs
using Statistics
using UUIDs

export NOISE_SCHEMA_VERSION, NOISE_MASTER_SEED, NOISE_LEVELS, NOISE_REPLICATES,
       NOISE_TEST_STEPS, NOISE_CONTROLLERS, normalize_protocol,
       normalize_controller, normalize_noise_level, level_tag, noise_seed,
       atomic_save, file_sha256, canonical_string, fingerprint,
       symbolize, source_defaults, select_sparse_sc_candidate,
       load_c_match_candidate, compute_channel_scales, protocol_test_cases,
       noisy_encoded_global_observation,
       build_protocol_manifest, load_protocol_manifest, worker_output_directory,
       episode_output_path, status_path, result_path, job_records

const NOISE_SCHEMA_VERSION = 1
const NOISE_MASTER_SEED = 20_260_901
const NOISE_LEVELS = (0.0, 0.01, 0.05, 0.10, 0.20)
const NOISE_REPLICATES = 10
const NOISE_TEST_STEPS = 200
const NOISE_CONTROLLERS = (:expert, :sparse, :c_match)

normalize_protocol(value)::Symbol = begin
    protocol = Symbol(lowercase(strip(string(value))))
    protocol in (:fixed, :varying) || throw(ArgumentError(
        "Protocol must be fixed or varying, got '$value'.",
    ))
    protocol
end

normalize_controller(value)::Symbol = begin
    controller = Symbol(lowercase(replace(strip(string(value)), "-" => "_")))
    controller in NOISE_CONTROLLERS || throw(ArgumentError(
        "Controller must be expert, sparse, or c_match, got '$value'.",
    ))
    controller
end

function normalize_noise_level(value)::Float64
    level = Float64(value)
    any(isequal(level), NOISE_LEVELS) || throw(ArgumentError(
        "Noise level must be one of $(join(NOISE_LEVELS, ", ")), got $value.",
    ))
    return level
end

function level_tag(level::Real)
    value = normalize_noise_level(level)
    return "n_" * replace(@sprintf("%.2f", value), "." => "p")
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

function noise_seed(protocol, level::Real, replicate::Integer, case_index::Integer)
    normalized_protocol = normalize_protocol(protocol)
    normalized_level = normalize_noise_level(level)
    normalized_level > 0 || throw(ArgumentError("The clean baseline has no noise seed."))
    1 <= replicate <= NOISE_REPLICATES || throw(ArgumentError("Replicate must be in 1:$NOISE_REPLICATES."))
    case_index >= 1 || throw(ArgumentError("case_index must be positive."))
    digest = SHA.sha256(codeunits(canonical_string((
        schema = NOISE_SCHEMA_VERSION,
        master_seed = NOISE_MASTER_SEED,
        protocol = normalized_protocol,
        noise_level = normalized_level,
        replicate = Int(replicate),
        case_index = Int(case_index),
    ))))
    value = UInt64(0)
    for byte in digest[1:8]
        value = (value << 8) | UInt64(byte)
    end
    return Int(mod(value, UInt64(2_000_000_000))) + 1
end

file_sha256(path::AbstractString) = open(path, "r") do io
    bytes2hex(SHA.sha256(io))
end

function atomic_save(path::AbstractString; entries...)
    mkpath(dirname(path))
    temporary = joinpath(dirname(path), ".$(basename(path)).$(getpid()).$(uuid4()).tmp")
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

symbolize(value::AbstractDict) = Dict{Symbol, Any}(Symbol(key) => item for (key, item) in value)
symbolize(value::NamedTuple) = Dict{Symbol, Any}(pairs(value))

function source_defaults(project_root::AbstractString)
    revision = joinpath(abspath(project_root), "Revision")
    return (
        package7_results = joinpath(revision, "Package7", "results", "260830_173924"),
        package8_results = joinpath(revision, "Package8", "results", "260830_231109"),
        go_study_results = joinpath(revision, "GO_Sensitivity", "results", "study"),
        baseline_results = joinpath(revision, "Baselines", "results"),
        distillation_root = joinpath(revision, "Expert_Apprentice_Distillation"),
    )
end

function require_file(path::AbstractString, description::AbstractString)
    isfile(path) || error("$description is missing: $path")
    return abspath(path)
end

function unique_file_named(root::AbstractString, filename::AbstractString)
    isdir(root) || error("Search directory is missing: $root")
    matches = String[]
    for (directory, _, files) in walkdir(root)
        filename in files && push!(matches, joinpath(directory, filename))
    end
    length(matches) == 1 || error(
        "Expected one '$filename' below '$root', found $(length(matches)).",
    )
    return abspath(only(matches))
end

function checkpoint_metadata(path::AbstractString, candidate_id)
    return JLD2.jldopen(path, "r") do file
        haskey(file, "candidate_metadata") || error("Candidate checkpoint has no metadata: $path")
        matches = filter(file["candidate_metadata"]) do raw
            metadata = symbolize(raw)
            string(metadata[:candidate_id]) == string(candidate_id)
        end
        length(matches) == 1 || error(
            "Expected one metadata record for candidate $candidate_id in '$path'.",
        )
        symbolize(only(matches))
    end
end

function selected_candidate_record(selection_path::AbstractString)
    require_file(selection_path, "Frozen selection")
    selection = JLD2.load(selection_path)
    Bool(selection["frozen_before_test"]) || error("Selection was not frozen before test: $selection_path")
    haskey(selection, "selection_uses_test_data") && Bool(selection["selection_uses_test_data"]) &&
        error("Selection used test data: $selection_path")
    candidate = symbolize(selection["candidate"])
    checkpoint_name = basename(replace(string(selection["checkpoint_path"]), '\\' => '/'))
    configuration_root = dirname(dirname(abspath(selection_path)))
    checkpoint_path = unique_file_named(configuration_root, checkpoint_name)
    checkpoint_hash = file_sha256(checkpoint_path)
    string(selection["checkpoint_sha256"]) == checkpoint_hash || error(
        "Relocated checkpoint hash mismatch for '$checkpoint_path'.",
    )
    metadata = haskey(candidate, :mask) ? candidate : merge(candidate, checkpoint_metadata(
        checkpoint_path,
        candidate[:candidate_id],
    ))
    return (;
        selection_path = abspath(selection_path),
        selection_sha256 = file_sha256(selection_path),
        checkpoint_path,
        checkpoint_sha256 = checkpoint_hash,
        candidate = metadata,
    )
end

function select_sparse_sc_candidate(protocol, package_results::AbstractString)
    normalized_protocol = normalize_protocol(protocol)
    expected_package = normalized_protocol === :fixed ? :package7_fixed_regularizer_comparison : :package8_varying_regularizer_comparison
    records = NamedTuple[]
    for configuration in ("go-sc", "gr-sc", "group-lasso-sc", "growl-sc")
        selection_path = joinpath(package_results, configuration, "analysis", "selected_test_candidate.jld2")
        selected = selected_candidate_record(selection_path)
        selection = JLD2.load(selection_path)
        Symbol(selection["experiment"]) === expected_package || error(
            "Unexpected experiment in '$selection_path'.",
        )
        candidate = selected.candidate
        endswith(string(candidate[:configuration]), "-sc") || error(
            "Sparse selection is not an SC candidate: $selection_path",
        )
        push!(records, (; configuration, selected...))
    end
    minimum_inputs = minimum(Int(record.candidate[:active_inputs]) for record in records)
    sparsest = filter(record -> Int(record.candidate[:active_inputs]) == minimum_inputs, records)
    chosen = first(sort(sparsest; by = record -> (
        Float64(record.candidate[:validation_matching]),
        string(record.candidate[:configuration]),
        string(record.candidate[:candidate_id]),
    )))
    audit = [(
        configuration = record.configuration,
        candidate_id = string(record.candidate[:candidate_id]),
        active_inputs = Int(record.candidate[:active_inputs]),
        active_groups = Int(record.candidate[:active_groups]),
        validation_matching = Float64(record.candidate[:validation_matching]),
        selected = record === chosen,
    ) for record in records]
    return merge(chosen, (
        selection_rule = :minimum_active_inputs_then_minimum_validation_mse,
        selection_audit = audit,
    ))
end

function go_checkpoint_path(go_study_results::AbstractString, protocol::Symbol, candidate)
    method = string(Symbol(candidate[:method]))
    strength = @sprintf("s%02d", Int(candidate[:strength_index]))
    replicate = @sprintf("r%02d", Int(candidate[:replicate]))
    filename = "checkpoint_" * lpad(string(Int(candidate[:update])), 12, '0') * ".jld2"
    return require_file(
        joinpath(go_study_results, string(protocol), method, strength, replicate, "candidates", filename),
        "C_match checkpoint",
    )
end

function load_c_match_candidate(protocol, go_study_results::AbstractString)
    normalized_protocol = normalize_protocol(protocol)
    manifest_path = require_file(
        joinpath(go_study_results, string(normalized_protocol), "analysis", "candidate_manifest.jld2"),
        "Package-6 candidate manifest",
    )
    manifest = JLD2.load(manifest_path)
    Bool(manifest["frozen_before_test"]) || error("Package-6 manifest was not frozen before test.")
    Bool(manifest["selection_uses_test_data"]) && error("Package-6 manifest used test data.")
    Symbol(manifest["protocol"]) === normalized_protocol || error("Package-6 protocol mismatch.")
    matches = filter(manifest["candidates"]) do raw
        candidate = symbolize(raw)
        Symbol(candidate[:selection_role]) === :C_match
    end
    length(matches) == 1 || error("Expected exactly one C_match candidate for $normalized_protocol.")
    candidate = symbolize(only(matches))
    checkpoint_path = go_checkpoint_path(go_study_results, normalized_protocol, candidate)
    metadata = checkpoint_metadata(checkpoint_path, candidate[:candidate_id])
    hydrated = merge(candidate, metadata)
    return (;
        selection_path = abspath(manifest_path),
        selection_sha256 = file_sha256(manifest_path),
        checkpoint_path,
        checkpoint_sha256 = file_sha256(checkpoint_path),
        expert_identifier = string(manifest["expert_identifier"]),
        candidate = hydrated,
    )
end

function scale_source_files(protocol, distillation_root::AbstractString)
    normalized_protocol = normalize_protocol(protocol)
    worker_root = joinpath(distillation_root, "worker_results")
    if normalized_protocol === :fixed
        return [require_file(joinpath(worker_root, "fixed", "fixed_shared.jld2"), "Fixed training corpus")]
    end
    directory = joinpath(worker_root, "varying", "train")
    isdir(directory) || error("Varying training corpus directory is missing: $directory")
    paths = sort!(filter(path -> endswith(lowercase(path), ".jld2"), readdir(directory; join = true)))
    length(paths) == 40 || error("Expected 40 Varying training shards, found $(length(paths)).")
    return abspath.(paths)
end

function compute_channel_scales(protocol, distillation_root::AbstractString)
    normalized_protocol = normalize_protocol(protocol)
    paths = scale_source_files(normalized_protocol, distillation_root)
    counts = zeros(Int, 3)
    sums = zeros(Float64, 3)
    sums_of_squares = zeros(Float64, 3)
    sources = NamedTuple[]
    for path in paths
        println("Reading channel scales from $path")
        observations, stored_protocol, stored_split, metadata = JLD2.jldopen(path, "r") do file
            result = haskey(file, "worker_result") ? symbolize(file["worker_result"]) : nothing
            getvalue(key::Symbol) = isnothing(result) ? file[string(key)] : result[key]
            (
                getvalue(:observations),
                Symbol(getvalue(:protocol)),
                Symbol(getvalue(:split)),
                symbolize(getvalue(:observation_metadata)),
            )
        end
        stored_protocol === normalized_protocol || error("Corpus protocol mismatch in '$path'.")
        normalized_protocol === :varying && stored_split !== :train && error("Non-training shard used for Varying scales: $path")
        size(observations)[1:3] == (3, 48, 8) || error("Unexpected observation shape in '$path'.")
        Bool(metadata[:joon_position_encoding]) || error("Expected Joon positional encoding in '$path'.")
        for horizontal_index in axes(observations, 2)
            correction = Float32(sin((2 * pi / size(observations, 2)) * horizontal_index))
            @views observations[1, horizontal_index, :, :] .-= correction
        end
        for channel in 1:3
            for raw in @view observations[channel, :, :, :]
                value = Float64(raw)
                counts[channel] += 1
                sums[channel] += value
                sums_of_squares[channel] += value * value
            end
        end
        push!(sources, (
            path = abspath(path),
            sha256 = file_sha256(path),
            split = stored_split,
            samples = size(observations, 4),
        ))
    end
    all(>(1), counts) || error("Insufficient samples for channel scales.")
    means = sums ./ counts
    variances = (sums_of_squares .- counts .* means .^ 2) ./ (counts .- 1)
    all(>(0), variances) || error("All physical channel variances must be positive.")
    return (
        protocol = normalized_protocol,
        definition = :sample_standard_deviation,
        channels = (:b, :w, :u),
        positional_encoding_removed = true,
        counts,
        means,
        scales = sqrt.(variances),
        sources,
    )
end

function noisy_encoded_global_observation(physical_observation, rng, level::Real, channel_scales)
    normalized_level = normalize_noise_level(level)
    size(physical_observation) == (3, 48, 8) || throw(DimensionMismatch(
        "Expected a physical 3×48×8 observation, got $(size(physical_observation)).",
    ))
    length(channel_scales) == 3 || throw(DimensionMismatch("Expected three channel scales."))
    all(scale -> isfinite(scale) && scale > 0, channel_scales) || throw(ArgumentError(
        "Channel scales must be finite and positive.",
    ))
    encoded = Float32.(physical_observation)
    standardized_noise = Array{Float32}(undef, size(encoded))
    randn!(rng, standardized_noise)
    for channel in axes(encoded, 1)
        @views encoded[channel, :, :] .+= Float32(normalized_level * channel_scales[channel]) .* standardized_noise[channel, :, :]
    end
    for horizontal_index in axes(encoded, 2)
        correction = Float32(sin((2 * pi / size(encoded, 2)) * horizontal_index))
        @views encoded[1, horizontal_index, :] .+= correction
    end
    return encoded
end

function protocol_test_cases(protocol, baseline_results::AbstractString)
    normalized_protocol = normalize_protocol(protocol)
    baseline_path = require_file(
        joinpath(baseline_results, string(normalized_protocol), "expert.jld2"),
        "Expert baseline",
    )
    baseline = JLD2.load(baseline_path)
    Symbol(baseline["protocol"]) === normalized_protocol || error("Baseline protocol mismatch.")
    Symbol(baseline["controller"]) === :expert || error("Expected an expert baseline.")
    episodes = baseline["episodes"]
    expected_count = normalized_protocol === :fixed ? 1 : 8
    length(episodes) == expected_count || error("Expected $expected_count baseline cases, found $(length(episodes)).")
    cases = [(
        index = index,
        case_id = string(episode.case_id),
        choice = episode.choice,
    ) for (index, episode) in enumerate(episodes)]
    return (;
        cases,
        baseline_path = abspath(baseline_path),
        baseline_sha256 = file_sha256(baseline_path),
        baseline_expert_sha256 = string(baseline["expert_sha256"]),
    )
end

function clean_source_for_sparse(protocol::Symbol, package_results::AbstractString, sparse)
    path = require_file(
        joinpath(package_results, sparse.configuration, "analysis", "test", "test_results.jld2"),
        "Sparse clean test result",
    )
    loaded = JLD2.load(path)
    string(loaded["candidate_id"]) == string(sparse.candidate[:candidate_id]) || error(
        "Sparse clean result does not match the selected candidate.",
    )
    Symbol(loaded["protocol"]) === protocol || error("Sparse clean result protocol mismatch.")
    return (
        path = abspath(path),
        sha256 = file_sha256(path),
        format = :package78_test_result,
        expert_identifier = string(loaded["expert_identifier"]),
    )
end

function clean_source_for_c_match(protocol::Symbol, go_study_results::AbstractString, candidate)
    cache_directory = joinpath(go_study_results, string(protocol), "analysis", "test", "cache")
    isdir(cache_directory) || error("Package-6 clean cache is missing: $cache_directory")
    matching_paths = String[]
    expert_identifiers = String[]
    for path in sort!(filter(name -> endswith(lowercase(name), ".jld2"), readdir(cache_directory; join = true)))
        JLD2.jldopen(path, "r") do file
            if haskey(file, "controller_id") && string(file["controller_id"]) == string(candidate[:candidate_id])
                push!(matching_paths, abspath(path))
                push!(expert_identifiers, string(file["expert_identifier"]))
            end
        end
    end
    expected = protocol === :fixed ? 1 : 8
    length(matching_paths) == expected || error(
        "Expected $expected C_match clean caches, found $(length(matching_paths)).",
    )
    return (
        paths = matching_paths,
        sha256 = file_sha256.(matching_paths),
        format = :package6_episode_cache,
        expert_identifiers = sort!(unique(expert_identifiers)),
    )
end

function controller_record(kind::Symbol, selected, clean_source)
    candidate = selected.candidate
    return (
        kind,
        label = kind === :sparse ? "Sparse SC apprentice" : "Package-6 C_match apprentice",
        controller_id = string(candidate[:candidate_id]),
        checkpoint_path = selected.checkpoint_path,
        checkpoint_sha256 = selected.checkpoint_sha256,
        selection_path = selected.selection_path,
        selection_sha256 = selected.selection_sha256,
        selection_role = kind === :sparse ? :sparsest_sc : :C_match,
        method = Symbol(candidate[:method]),
        configuration = kind === :sparse ? string(candidate[:configuration]) : "go-sc",
        active_inputs = Int(candidate[:active_inputs]),
        active_groups = Int(candidate[:active_groups]),
        validation_matching = Float64(candidate[:validation_matching]),
        input_mask = Float32.(candidate[:mask]),
        clean_source,
    )
end

function job_records(protocol)
    normalized_protocol = normalize_protocol(protocol)
    return [(
        protocol = normalized_protocol,
        controller,
        noise_level = level,
        level_tag = level_tag(level),
        replicate_count = level == 0 ? 0 : NOISE_REPLICATES,
        id = "p10_$(normalized_protocol)_$(controller)_$(level_tag(level))",
    ) for level in NOISE_LEVELS for controller in NOISE_CONTROLLERS]
end

function build_protocol_manifest(
    output::AbstractString,
    protocol;
    package_results::AbstractString,
    go_study_results::AbstractString,
    baseline_results::AbstractString,
    distillation_root::AbstractString,
    experiment_id::AbstractString,
)
    normalized_protocol = normalize_protocol(protocol)
    package_root = abspath(package_results)
    sparse = select_sparse_sc_candidate(normalized_protocol, package_root)
    c_match = load_c_match_candidate(normalized_protocol, abspath(go_study_results))
    test_data = protocol_test_cases(normalized_protocol, abspath(baseline_results))
    scales = compute_channel_scales(normalized_protocol, abspath(distillation_root))
    expert_path = require_file(
        joinpath(distillation_root, "experts", string(normalized_protocol), "agent.jld2"),
        "Expert checkpoint",
    )
    expert_hash = file_sha256(expert_path)
    expert_identifier = "sha256:$expert_hash"
    c_match.expert_identifier == expert_identifier || error("C_match expert identity mismatch.")
    test_data.baseline_expert_sha256 == expert_hash || error("Expert baseline checkpoint identity mismatch.")
    expert_clean = (
        path = test_data.baseline_path,
        sha256 = test_data.baseline_sha256,
        format = :revision_expert_baseline,
    )
    sparse_clean = clean_source_for_sparse(normalized_protocol, package_root, sparse)
    c_match_clean = clean_source_for_c_match(normalized_protocol, abspath(go_study_results), c_match.candidate)
    sparse_clean.expert_identifier == expert_identifier || error("Sparse clean-result expert identity mismatch.")
    c_match_clean.expert_identifiers == [expert_identifier] || error("C_match clean-cache expert identity mismatch.")
    controllers = [
        (
            kind = :expert,
            label = "Dense MAT expert",
            controller_id = "expert",
            checkpoint_path = expert_path,
            checkpoint_sha256 = expert_hash,
            selection_path = "",
            selection_sha256 = "",
            selection_role = :expert,
            method = :mat,
            configuration = "dense",
            active_inputs = 1152,
            active_groups = 96,
            validation_matching = 0.0,
            input_mask = Float32[],
            clean_source = expert_clean,
        ),
        controller_record(:sparse, sparse, sparse_clean),
        controller_record(:c_match, c_match, c_match_clean),
    ]
    length(unique(record.controller_id for record in controllers)) == 3 || error("Controller IDs are not unique.")
    jobs = job_records(normalized_protocol)
    identity = (
        schema_version = NOISE_SCHEMA_VERSION,
        experiment = :package10_sensor_noise,
        experiment_id = String(experiment_id),
        protocol = normalized_protocol,
        expert_identifier,
        noise_model = :additive_iid_zero_mean_gaussian,
        scale_definition = scales.definition,
        noise_levels = collect(NOISE_LEVELS),
        replicate_count = NOISE_REPLICATES,
        test_steps = NOISE_TEST_STEPS,
        master_seed = NOISE_MASTER_SEED,
        cases = test_data.cases,
        controllers,
        channel_scales = scales,
        jobs,
    )
    manifest_hash = fingerprint(identity)
    atomic_save(
        output;
        identity...,
        manifest_fingerprint = manifest_hash,
        frozen_before_noise_test = true,
        selection_uses_noise_results = false,
        created_at = string(Dates.now(Dates.UTC)),
    )
    return (path = abspath(output), fingerprint = manifest_hash, controllers, jobs, scales)
end

function load_protocol_manifest(path::AbstractString)
    require_file(path, "Noise-study manifest")
    loaded = JLD2.load(path)
    Int(loaded["schema_version"]) == NOISE_SCHEMA_VERSION || error("Noise manifest schema mismatch.")
    Symbol(loaded["experiment"]) === :package10_sensor_noise || error("Unexpected noise manifest experiment.")
    Bool(loaded["frozen_before_noise_test"]) || error("Noise manifest is not frozen.")
    Bool(loaded["selection_uses_noise_results"]) && error("Noise results influenced controller selection.")
    return symbolize(loaded)
end

worker_output_directory(results_root::AbstractString, experiment_id, protocol, controller, level) = joinpath(
    abspath(results_root), String(experiment_id), string(normalize_protocol(protocol)),
    string(normalize_controller(controller)), level_tag(level),
)

function episode_output_path(output::AbstractString, level::Real, replicate::Integer, case_id::AbstractString)
    replicate_directory = normalize_noise_level(level) == 0 ? "baseline" : @sprintf("r%02d", replicate)
    safe_case = replace(case_id, r"[^A-Za-z0-9_-]" => "_")
    return joinpath(output, "episodes", replicate_directory, "$safe_case.jld2")
end

status_path(output::AbstractString) = joinpath(output, "status.jld2")
result_path(output::AbstractString) = joinpath(output, "result.jld2")

end
