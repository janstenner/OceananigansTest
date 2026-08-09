using Zygote
using Optimisers
using Flux
using LinearAlgebra
using Random
using Statistics




const EXPERT_APPRENTICE_PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const EXPERT_APPRENTICE_PROTOCOL = Symbol(
    lowercase(get(ENV, "DISTILLATION_PROTOCOL", "varying")),
)
EXPERT_APPRENTICE_PROTOCOL in (:fixed, :varying) || error(
    "DISTILLATION_PROTOCOL must be fixed or varying.",
)

randomIC = EXPERT_APPRENTICE_PROTOCOL === :varying
group_channels = lowercase(get(ENV, "DISTILLATION_GROUP_CHANNELS", "true")) in
                 ("1", "true", "yes")

revision_run_file = randomIC ? "VaryingIC_MAT.jl" : "FixedIC_MAT.jl"
include(joinpath(
    EXPERT_APPRENTICE_PROJECT_ROOT,
    "Revision",
    "Run_Files",
    revision_run_file,
))
if !isdefined(@__MODULE__, :DISTILLATION_CORPUS)
    include(joinpath(@__DIR__, "DistillationCorpus.jl"))
end
if !isdefined(@__MODULE__, :ParetoArchiveManager)
    include(joinpath(@__DIR__, "ParetoArchive.jl"))
end

allow_fresh_expert = lowercase(
    get(ENV, "DISTILLATION_ALLOW_FRESH_EXPERT", "false"),
) in ("1", "true", "yes")
const DISTILLATION_EXPERT_METADATA = load_distillation_expert!(
    EXPERT_APPRENTICE_PROTOCOL;
    allow_fresh_expert,
)




group_rows_by_overlap = true






customCrossAttention = true
jointPPO = false
one_by_one_training = false
positional_encoding = 3 #ZeroEncoding

joon_pe = true
new_pe = false
square_rewards = false



if randomIC
    block_num = 1
    dim_model = 44
    head_num = 2
    head_dim = 22
    ffn_dim = 44
    drop_out = 0.00#1

    learning_rate = 2e-4
    clip_grad = Inf
else
    block_num = 1
    dim_model = 32
    head_num = 2
    head_dim = 16
    ffn_dim = 32
    drop_out = 0.00#1

    learning_rate = 2e-4
    clip_grad = Inf
end


betas = (0.9, 0.999)

apprentice_training_kind = :growl
apprentice_training_rIC = randomIC

const APPRENTICE_KIND_CONFIG = Dict{Symbol, NamedTuple}(
    :go => (
        label = "Group Ordered",
        regularizer = :grouped,
        regularization_strength_fixed = 0.09,
        regularization_strength_varying = 0.025,
        uses_operator_weights = false,
        theta_mode = :go,
    ),
    :gr => (
        label = "Group Reweighted",
        regularizer = :group_reweighted,
        regularization_strength_fixed = 0.00004,
        regularization_strength_varying = 0.0001,
        uses_operator_weights = true,
        theta_mode = nothing,
    ),
    :group_lasso => (
        label = "Group Lasso",
        regularizer = :grouped,
        regularization_strength_fixed = 0.0001,
        regularization_strength_varying = 0.00012,
        uses_operator_weights = false,
        theta_mode = :group_lasso,
    ),
    :growl => (
        label = "GrOWL",
        regularizer = :grouped,
        regularization_strength_fixed = 0.00006,
        regularization_strength_varying = 0.0004,
        uses_operator_weights = false,
        theta_mode = :growl,
    ),
)

function normalize_apprentice_kind(kind)::Symbol
    kind_sym = kind isa Symbol ? kind : Symbol(lowercase(string(kind)))
    haskey(APPRENTICE_KIND_CONFIG, kind_sym) || error(
        "Unknown apprentice kind '$kind_sym'. Available kinds: " *
        join(string.(available_apprentice_kinds()), ", "),
    )
    return kind_sym
end

function apprentice_kind_sort_key(kind)::Tuple{Int, String}
    kind_sym = kind isa Symbol ? kind : Symbol(lowercase(string(kind)))
    priority = Dict(:go => 0, :gr => 1, :group_lasso => 2, :growl => 3)
    return (get(priority, kind_sym, 100), string(kind_sym))
end

function available_apprentice_kinds()
    kinds = collect(keys(APPRENTICE_KIND_CONFIG))
    sort!(kinds, by = apprentice_kind_sort_key)
    return kinds
end

function apprentice_kind_label(kind)::String
    normalized = normalize_apprentice_kind(kind)
    config = get(APPRENTICE_KIND_CONFIG, normalized, nothing)
    if config === nothing
        return replace(string(normalized), "_" => " ")
    end
    return config.label
end

function apprentice_kind_config(kind)
    normalized = normalize_apprentice_kind(kind)
    config = get(APPRENTICE_KIND_CONFIG, normalized, nothing)
    config === nothing && error(
        "Unknown apprentice kind '$normalized'. Add it to APPRENTICE_KIND_CONFIG. " *
        "Available kinds: $(join(string.(available_apprentice_kinds()), ", "))."
    )
    return normalized, config
end

Base.@kwdef struct ApprenticeTrainingConfig
    regularized_updates::Int = randomIC ? 15_000 : 9_000
    post_pruning_finetune_updates::Int = 0
    batch_size::Int = randomIC ? 100 : 20
    proximal_interval::Int = 1
    reweight_interval::Int = 10
    regularization_strength::Float64 = NaN
    validation_batch_size::Int = 256
    validation_prediction_mode::Symbol = :autoregressive
    diagnostic_teacher_forced::Bool = true

    function ApprenticeTrainingConfig(
        regularized_updates::Int,
        post_pruning_finetune_updates::Int,
        batch_size::Int,
        proximal_interval::Int,
        reweight_interval::Int,
        regularization_strength::Float64,
        validation_batch_size::Int,
        validation_prediction_mode::Symbol,
        diagnostic_teacher_forced::Bool,
    )
        regularized_updates >= 0 || throw(ArgumentError("regularized_updates must be nonnegative."))
        post_pruning_finetune_updates >= 0 || throw(ArgumentError("post_pruning_finetune_updates must be nonnegative."))
        batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
        proximal_interval > 0 || throw(ArgumentError("proximal_interval must be positive."))
        reweight_interval > 0 || throw(ArgumentError("reweight_interval must be positive."))
        validation_batch_size > 0 || throw(ArgumentError("validation_batch_size must be positive."))
        validation_prediction_mode in (:autoregressive, :teacher_forced) || throw(
            ArgumentError("validation_prediction_mode must be :autoregressive or :teacher_forced."),
        )
        return new(
            regularized_updates,
            post_pruning_finetune_updates,
            batch_size,
            proximal_interval,
            reweight_interval,
            regularization_strength,
            validation_batch_size,
            validation_prediction_mode,
            diagnostic_teacher_forced,
        )
    end
end

function resolved_regularization_strength(config::ApprenticeTrainingConfig, kind_config; rIC::Bool = randomIC)
    isfinite(config.regularization_strength) && return config.regularization_strength
    return rIC ? kind_config.regularization_strength_varying : kind_config.regularization_strength_fixed
end

apprentice_agent = create_agent_mat(n_actors = actuators,
                    action_space = actionspace,
                    state_space = env.state_space,
                    use_gpu = false, 
                    rng = rng,
                    y = y, p = p,
                    start_steps = start_steps, 
                    start_policy = start_policy,
                    update_freq = update_freq,
                    learning_rate = learning_rate,
                    nna_scale = 1.0,
                    nna_scale_critic = 1.0,
                    drop_middle_layer = true,
                    drop_middle_layer_critic = true,
                    fun = gelu,
                    clip1 = false,
                    n_epochs = n_epochs,
                    n_microbatches = n_microbatches,
                    logσ_is_network = false,
                    max_σ = max_σ,
                    entropy_loss_weight = entropy_loss_weight,
                    adaptive_weights = false,
                    clip_grad = clip_grad,
                    target_kl = target_kl,
                    start_logσ = -10.0f0,
                    dim_model = dim_model,
                    block_num = block_num,
                    head_num = head_num,
                    head_dim = head_dim,
                    ffn_dim = ffn_dim,
                    drop_out = drop_out,
                    betas = betas,
                    jointPPO = jointPPO,
                    customCrossAttention = customCrossAttention,
                    one_by_one_training = one_by_one_training,
                    clip_range = clip_range,
                    tanh_end = tanh_end,
                    positional_encoding = positional_encoding,
                    )


apprentice = apprentice_agent.policy

encoder = apprentice.encoder
decoder = apprentice.decoder

# Training, validation, and candidate generation are defined below from the
# DistillationCorpus data. No mutable global input mask is used.
function get_row_groups(;group_channels = true)

    row_groups = []

    index_array = collect(1:size(env.state[:,1])[1])

    if new_pe
        channel_size = sensors[2]+1
    else
        channel_size = sensors[2]
    end

    index_y = reshape(index_array, 3,window_size,channel_size)

    # create stencil for grouping
    center_point = Int(ceil(window_size/2))
    agent_delta = Int(sensors[1] / actuators)


    anchor_steps = Int(ceil(center_point/agent_delta)+1)
    
    if group_channels
        stencil_index_array = collect(1:agent_delta*(channel_size))
        index_stencil = reshape(stencil_index_array, 1, agent_delta, channel_size)

        anchors = [
            [1,2,3],
            [center_point + (j * agent_delta) for j in -anchor_steps:anchor_steps],
            [1]
        ]
    else
        stencil_index_array = collect(1:3*agent_delta*(channel_size))
        index_stencil = reshape(stencil_index_array, 3, agent_delta, channel_size)

        anchors = [
            [1],
            [center_point + (j * agent_delta) for j in -anchor_steps:anchor_steps],
            [1]
        ]
    end
    
    for i in stencil_index_array
        # get the stencil offset for the current index
        stencil_offset = collect(findfirst(x -> x == i, index_stencil).I .- 1)

        # get the anchor points for the current stencil offset
        anchor_points = deepcopy(anchors)
        anchor_points[1] .+=  stencil_offset[1]
        anchor_points[2] .+=  stencil_offset[2]
        anchor_points[3] .+=  stencil_offset[3]

        # filter for valid indices
        anchor_points[1] = filter(i -> (1 ≤ i ≤ size(index_y, 1)), anchor_points[1])
        anchor_points[2] = filter(i -> (1 ≤ i ≤ size(index_y, 2)), anchor_points[2])
        anchor_points[3] = filter(i -> (1 ≤ i ≤ size(index_y, 3)), anchor_points[3])

        # get the indices of the row group
        push!(row_groups, index_y[anchor_points...][:])
    end

    return row_groups
end

row_groups = get_row_groups(group_channels = group_channels)


# utility function to check for duplicates of row_groups. Should return false
function any_shared(c)
    seen = Set{Int}()

    for arr in c
        for x in arr
            if x in seen
                return true              # x appeared in a previous sub‐array
            end
            push!(seen, x)
        end
    end
    return false                        # no element was seen twice
end

function build_theta_is(n_groups::Int, theta_mode::Symbol)
    n_groups > 0 || error("n_groups must be positive, got $n_groups")

    if theta_mode == :go
        return Float64[(i - 1) / n_groups for i in 1:n_groups]
    elseif theta_mode == :group_lasso
        return ones(Float64, n_groups)
    elseif theta_mode == :growl
        return Float64[(i - 1) / n_groups for i in n_groups:-1:1]
    else
        error("Unsupported theta_mode '$theta_mode'. Use :go, :group_lasso, or :growl.")
    end
end




function apply_grouped_regularizer!(
    model_weights;
    groups,
    regularization_strength::Real,
    theta_mode::Symbol,
)
    reshaped_weight = transpose(model_weights)

    # Compute the L2 norm for each row.
    n_groups = length(groups)
    n2_groups = [norm(reshaped_weight[i, :][:], 2) for i in groups]

    # --- Create GrOWL parameters ---
    theta_is = build_theta_is(n_groups, theta_mode)
    theta_is .*= regularization_strength

    # Apply the proximal operator.
    new_n2_groups = proxOWL(deepcopy(n2_groups), deepcopy(theta_is))

    # --- Rescale the weight rows ---
    new_W = similar(reshaped_weight)
    eps_val = eps(Float32)

    for i in 1:n_groups

        if new_n2_groups[i] < eps_val
            # If the norm is too small, set all rows belonging to the group to zero.
            for j in groups[i]
                new_W[j, :] .= zeros(eltype(reshaped_weight), size(reshaped_weight, 2))
            end
        else
            # Scale all rows belonging to the group.
            for j in groups[i]
                new_W[j, :] .= reshaped_weight[j, :] .* (new_n2_groups[i] / n2_groups[i])
            end       
        end
    end


    model_weights .= transpose(new_W)
    return model_weights
end



function apply_group_reweighted_regularizer!(
    model_weights;
    groups,
    operator_weights::AbstractVector,
    regularization_strength::Real,
)
    reshaped_weight = transpose(model_weights)

    # Compute the L2 norm for each row group.
    n_groups = length(groups)
    n2_groups = [norm(reshaped_weight[i, :][:], 2) for i in groups]

    length(operator_weights) == n_groups || error("operator_weights length must match number of groups.")

    # Apply weighted L1 proximal operator.
    new_n2_groups = prox_weighted_l1(
        deepcopy(n2_groups),
        deepcopy(operator_weights .* regularization_strength),
    )

    # --- Rescale the weight rows ---
    new_W = similar(reshaped_weight)
    eps_val = eps(Float32)

    for i in 1:n_groups
        if new_n2_groups[i] < eps_val
            # If the norm is too small, set all rows belonging to the group to zero.
            for j in groups[i]
                new_W[j, :] .= zeros(eltype(reshaped_weight), size(reshaped_weight, 2))
            end
        else
            # Scale all rows belonging to the group.
            for j in groups[i]
                new_W[j, :] .= reshaped_weight[j, :] .* (new_n2_groups[i] / n2_groups[i])
            end
        end
    end


    model_weights .= transpose(new_W)
    return model_weights
end



function prox_weighted_l1(z::Vector, mu::Vector)
    length(z) == length(mu) || error("z and mu must have the same length.")
    x = z .- mu
    x = max.(x, zero(eltype(x)))
    return x
end



function proxOWL(z::Vector, mu::Vector)
    # store the signs of z.
    sgn = sign.(z)
    # Work with absolute values.
    z_abs = abs.(z)
    # Sort z_abs in non-increasing (descending) order.
    indx = sortperm(z_abs, rev=true)
    z_sorted = z_abs[indx]
    n = length(z_sorted)
    x = zeros(n)
    diff = z_sorted .- mu
    # Reverse diff to mimic Python’s diff[::-1]
    diff_rev = reverse(diff)
    # Find the first index in the reversed diff that is > 0.
    indc = findfirst(x -> x > 0, diff_rev)
    flag = indc === nothing ? 0.0 : diff_rev[indc]
    if flag > 0
        # In Python: k = n - indc, but note the 1-index adjustment in Julia.
        k = n - indc + 1
        v1 = deepcopy(z_sorted[1:k])
        v2 = deepcopy(mu[1:k])
        v = proxOWL_segments(v1, v2)
        # Prepare an output array in original order.
        x_orig = zeros(n)
        for j in 1:k
            # indx[j] holds the original index for the j-th largest element.
            x_orig[indx[j]] = v[j]
        end
        x = x_orig
    end
    # Restore original signs.
    x = sgn .* x
    return x
end



function proxOWL_segments(A::Vector, B::Vector)
    modified = true
    k = 0
    max_its = 1000
    # Loop until no modifications occur or we exceed the maximum iterations.
    while modified && k <= max_its
        modified = false
        segments = Tuple{Int,Int}[]
        new_start = true
        start_idx = nothing
        end_idx = nothing

        for i in 1:length(A)-1
            if (A[i] - B[i] < A[i+1] - B[i+1])
                modified = true
                if new_start
                    start_idx = i
                    new_start = false
                end
                continue
            elseif (A[i] - B[i] >= A[i+1] - B[i+1])
                if start_idx !== nothing
                    end_idx = i
                    push!(segments, (start_idx, end_idx))
                end
                new_start = true
                start_idx = nothing
                end_idx = nothing
            end
        end

        # If a segment was started but not ended, finish it.
        if (start_idx !== nothing) && (end_idx === nothing)
            end_idx = length(A)
            push!(segments, (start_idx, end_idx))
        end

        # If no segments were found, exit the loop.
        if isempty(segments)
            break
        end

        # For each segment, replace A and B over that range with their means.
        for (s, e) in segments
            avg_A = mean(A[s:e])
            avg_B = mean(B[s:e])
            for j in s:e
                A[j] = avg_A
                B[j] = avg_B
            end
            modified = true
        end
        k += 1
    end

    # Compute X = A - B and set any negative values to zero.
    X = A .- B
    X = map(x -> x < 0 ? 0.0 : x, X)
    return X
end


struct HardThresholdSpec
    id::Symbol
    mode::Symbol
    value::Float64
    analysis_scope::Symbol

    function HardThresholdSpec(
        id::Symbol,
        mode::Symbol,
        value::Real;
        analysis_scope::Symbol = :later_packages,
    )
        mode in (:absolute, :relative_to_max, :keep_groups) || throw(
            ArgumentError("Hard-threshold mode must be :absolute, :relative_to_max, or :keep_groups."),
        )
        value >= 0 || throw(
            ArgumentError("Hard-threshold values must be nonnegative."),
        )
        return new(id, mode, Float64(value), analysis_scope)
    end
end

function regularizer_groups(model; group_rows_by_overlap::Bool, group_channels::Bool)
    n_inputs = size(transpose(model.encoder.embedding.weight), 1)
    groups = group_rows_by_overlap ? get_row_groups(; group_channels) : [[index] for index in 1:n_inputs]
    any_shared(groups) && error("Regularizer groups overlap; masks would no longer be unambiguous.")
    sort(vcat(groups...)) == collect(1:n_inputs) || error("Regularizer groups do not cover every input.")
    return groups
end

function group_importances(model, groups)
    weights_by_input = transpose(model.encoder.embedding.weight)
    return Float64[norm(view(weights_by_input, group, :)) for group in groups]
end

function local_input_mask(groups, active_groups, n_inputs::Integer)
    length(groups) == length(active_groups) || throw(DimensionMismatch("One activity flag is required per group."))
    result = zeros(Float32, n_inputs)
    for (group, active) in zip(groups, active_groups)
        active && (result[group] .= 1.0f0)
    end
    return result
end

function hard_threshold_group_mask(importances, spec::HardThresholdSpec)
    if spec.mode === :absolute
        return importances .> spec.value
    elseif spec.mode === :relative_to_max
        maximum_importance = maximum(importances; init = 0.0)
        cutoff = spec.value * maximum_importance
        return importances .> cutoff
    end

    keep_count = clamp(round(Int, spec.value), 0, length(importances))
    active = falses(length(importances))
    keep_count == 0 && return active
    ordering = sortperm(importances; rev = true)
    active[ordering[1:keep_count]] .= true
    return active
end

function global_input_mask(
    local_mask;
    actuator_sensor_indices = actuators_to_sensors,
    horizontal_sensor_count::Int = sensors[1],
    vertical_sensor_count::Int = sensors[2],
    local_window_size::Int = window_size,
)
    expected_inputs = 3 * local_window_size * vertical_sensor_count
    length(local_mask) == expected_inputs || throw(DimensionMismatch(
        "Expected a local mask with $expected_inputs entries, got $(length(local_mask)).",
    ))
    local_tensor = reshape(Bool.(local_mask), 3, local_window_size, vertical_sensor_count)
    result = falses(3, horizontal_sensor_count, vertical_sensor_count)
    half_width = fld(local_window_size, 2)
    for sensor_index in actuator_sensor_indices
        horizontal_indices = [
            mod1(Int(sensor_index) + offset, horizontal_sensor_count)
            for offset in -half_width:half_width
        ]
        result[:, horizontal_indices, :] .|= local_tensor
    end
    return result
end

function candidate_masks(model, threshold_specs; groups)
    importances = group_importances(model, groups)
    n_inputs = size(transpose(model.encoder.embedding.weight), 1)
    specifications = collect(threshold_specs)
    length(unique(spec.id for spec in specifications)) == length(specifications) || error(
        "Hard-threshold IDs must be unique.",
    )

    definitions = [(
        threshold_id = :native,
        threshold_kind = :native,
        pareto_scope = :native,
        threshold_mode = :exact_zero,
        threshold_value = 0.0,
        analysis_scope = :package6_native_sensitivity,
        active_groups = importances .> 0.0,
    )]
    for spec in specifications
        spec.id === :native && error(":native is reserved for the unmodified native-sparsity candidate.")
        push!(definitions, (
            threshold_id = spec.id,
            threshold_kind = :hard_threshold,
            pareto_scope = :hard_threshold,
            threshold_mode = spec.mode,
            threshold_value = spec.value,
            analysis_scope = spec.analysis_scope,
            active_groups = hard_threshold_group_mask(importances, spec),
        ))
    end

    return map(definitions) do definition
        local_mask = local_input_mask(groups, definition.active_groups, n_inputs)
        global_mask = global_input_mask(local_mask)
        active_sensor_locations = count(dropdims(any(global_mask; dims = 1); dims = 1))
        Dict{Symbol, Any}(
            :threshold_id => definition.threshold_id,
            :threshold_kind => definition.threshold_kind,
            :pareto_scope => definition.pareto_scope,
            :threshold_mode => definition.threshold_mode,
            :threshold_value => definition.threshold_value,
            :analysis_scope => definition.analysis_scope,
            :mask => local_mask,
            :group_mask => BitVector(definition.active_groups),
            :global_mask => BitArray(global_mask),
            :active_groups => count(definition.active_groups),
            :active_inputs => count(global_mask),
            :active_sensor_locations => active_sensor_locations,
            :group_importances => copy(importances),
        )
    end
end

function teacher_forced_actions(model, observations, expert_actions)
    observation_representation, _ = model.encoder(observations)
    action_dimension = size(model.decoder.embedding.weight, 2)
    batch_size = size(observations, 3)
    shifted_actions = cat(
        zeros(Float32, action_dimension, 1, batch_size),
        expert_actions[:, 1:end-1, :];
        dims = 2,
    )
    means, _ = model.decoder(shifted_actions, observation_representation)
    return means
end

function apprentice_actions(model, observations, expert_actions; prediction_mode::Symbol)
    if prediction_mode === :autoregressive
        return prob(model, observations, nothing).μ
    elseif prediction_mode === :teacher_forced
        return teacher_forced_actions(model, observations, expert_actions)
    end
    throw(ArgumentError("prediction_mode must be :autoregressive or :teacher_forced."))
end

function evaluate_expert_matching(
    model,
    dataset,
    input_mask;
    batch_size::Int = 256,
    prediction_mode::Symbol = :autoregressive,
)
    sample_count = Int(dataset[:sample_count])
    sample_count > 0 || error("Cannot validate against an empty distillation dataset.")
    length(input_mask) == size(env.state, 1) || throw(DimensionMismatch("Input mask has the wrong length."))
    broadcast_mask = reshape(Float32.(input_mask), :, 1, 1)
    squared_error_sum = 0.0
    element_count = 0

    for first_index in 1:batch_size:sample_count
        last_index = min(first_index + batch_size - 1, sample_count)
        batch = distillation_batch(
            dataset,
            first_index:last_index;
            actuator_sensor_indices = actuators_to_sensors,
            window_size,
        )
        observations = batch.observations .* broadcast_mask
        predictions = apprentice_actions(
            model,
            observations,
            batch.expert_actions;
            prediction_mode,
        )
        difference = predictions .- batch.expert_actions
        all(isfinite, difference) || return Inf
        squared_error_sum += sum(abs2, difference)
        element_count += length(difference)
    end
    return squared_error_sum / element_count
end

function evaluate_candidate_checkpoint!(
    manager::ParetoArchiveManager,
    model,
    update::Integer,
    validation_dataset,
    threshold_specs;
    groups,
    method::Symbol,
    regularization_strength::Real,
    validation_batch_size::Int = 256,
    prediction_mode::Symbol = :autoregressive,
    diagnostic_teacher_forced::Bool = true,
)
    records = candidate_masks(model, threshold_specs; groups)
    for record in records
        record[:validation_matching] = evaluate_expert_matching(
            model,
            validation_dataset,
            record[:mask];
            batch_size = validation_batch_size,
            prediction_mode,
        )
        record[:validation_prediction_mode] = prediction_mode
        if diagnostic_teacher_forced
            record[:teacher_forced_validation_matching] = evaluate_expert_matching(
                model,
                validation_dataset,
                record[:mask];
                batch_size = validation_batch_size,
                prediction_mode = :teacher_forced,
            )
        end
        record[:numeric_status] = isfinite(record[:validation_matching]) ? :ok : :nonfinite
        record[:method] = method
        record[:regularization_strength] = Float64(regularization_strength)
    end
    return record_candidate_batch!(
        manager,
        update,
        records;
        model_payload = deepcopy(model),
        evaluation_metadata = Dict{Symbol, Any}(
            :dataset_source_files => copy(validation_dataset[:source_files]),
            :sample_count => Int(validation_dataset[:sample_count]),
            :prediction_mode => prediction_mode,
        ),
    )
end

mutable struct DistillationBatchSampler
    order::Vector{Int}
    cursor::Int
    epoch::Int
end

function DistillationBatchSampler(sample_count::Integer, rng::AbstractRNG)
    sample_count > 0 || error("Cannot train from an empty distillation dataset.")
    return DistillationBatchSampler(randperm(rng, Int(sample_count)), 1, 1)
end

function next_batch_indices!(sampler::DistillationBatchSampler, batch_size::Integer, rng::AbstractRNG)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive."))
    result = Int[]
    while length(result) < batch_size
        available = length(sampler.order) - sampler.cursor + 1
        take = min(batch_size - length(result), available)
        append!(result, sampler.order[sampler.cursor:sampler.cursor + take - 1])
        sampler.cursor += take
        if sampler.cursor > length(sampler.order)
            sampler.order = randperm(rng, length(sampler.order))
            sampler.cursor = 1
            sampler.epoch += 1
        end
    end
    return result
end

function model_is_finite(model)
    return all(parameter -> all(isfinite, parameter), Flux.trainables(model))
end

function update_operator_weights!(operator_weights, model, groups)
    importances = group_importances(model, groups)
    epsilon = convert(eltype(operator_weights), 1e-3)
    operator_weights .= one(eltype(operator_weights)) ./ (importances .+ epsilon)
    return operator_weights
end

function stop_on_numerical_failure!(
    archive_manager,
    update::Integer,
    message::AbstractString;
    model,
    sampler,
    training_rng,
    losses,
    input_mask,
    operator_weights,
)
    if !isnothing(archive_manager)
        save_resume_checkpoint!(
            archive_manager,
            update,
            Dict{Symbol, Any}(
                :model => deepcopy(model),
                :sampler => deepcopy(sampler),
                :rng => deepcopy(training_rng),
                :losses => copy(losses),
                :finetune_input_mask => copy(input_mask),
                :operator_weights => copy(operator_weights),
                :failure_message => String(message),
            );
            status = :failed,
        )
    end
    error(message)
end

function train_apprentice!(
    model = apprentice;
    method::Symbol = apprentice_training_kind,
    train_dataset = distillation_dataset(EXPERT_APPRENTICE_PROTOCOL, :train),
    validation_dataset = distillation_dataset(EXPERT_APPRENTICE_PROTOCOL, :validation),
    config::ApprenticeTrainingConfig = ApprenticeTrainingConfig(),
    archive_manager::Union{Nothing, ParetoArchiveManager} = nothing,
    threshold_specs = HardThresholdSpec[],
    group_rows_by_overlap::Bool = true,
    group_channels::Bool = group_channels,
    training_rng::AbstractRNG = rng,
    resume::Bool = false,
)
    method, method_config = apprentice_kind_config(method)
    regularization_strength = resolved_regularization_strength(
        config,
        method_config;
        rIC = EXPERT_APPRENTICE_PROTOCOL === :varying,
    )
    groups = regularizer_groups(model; group_rows_by_overlap, group_channels)
    sample_count = Int(train_dataset[:sample_count])
    sampler = DistillationBatchSampler(sample_count, training_rng)
    operator_weights = ones(eltype(model.encoder.embedding.weight), length(groups))
    input_mask = ones(Float32, size(env.state, 1))
    losses = Float64[]
    total_updates = config.regularized_updates + config.post_pruning_finetune_updates
    first_update = 1

    if resume
        isnothing(archive_manager) && error("resume=true requires an archive_manager.")
        checkpoint = load_resume_checkpoint(archive_manager)
        isnothing(checkpoint) && error("No resume/latest.jld2 exists for this run.")
        checkpoint.status === :complete && error("The requested run is already complete.")
        state = checkpoint.resume_state
        model = state[:model]
        sampler = state[:sampler]
        training_rng = state[:rng]
        losses = Float64.(state[:losses])
        input_mask = Float32.(state[:finetune_input_mask])
        groups = regularizer_groups(model; group_rows_by_overlap, group_channels)
        operator_weights = copy(state[:operator_weights])
        first_update = checkpoint.update + 1
    end

    global apprentice_training_kind = method
    global apprentice_training_rIC = EXPERT_APPRENTICE_PROTOCOL === :varying

    if first_update == 1 && !isnothing(archive_manager) && should_evaluate_candidates(archive_manager.schedule, 0)
        evaluate_candidate_checkpoint!(
            archive_manager,
            model,
            0,
            validation_dataset,
            threshold_specs;
            groups,
            method,
            regularization_strength,
            validation_batch_size = config.validation_batch_size,
            prediction_mode = config.validation_prediction_mode,
            diagnostic_teacher_forced = config.diagnostic_teacher_forced,
        )
    end

    for update in first_update:total_updates
        if update == config.regularized_updates + 1
            native_candidate = only(filter(
                candidate -> candidate[:threshold_id] === :native,
                candidate_masks(model, HardThresholdSpec[]; groups),
            ))
            input_mask = native_candidate[:mask]
        end

        indices = next_batch_indices!(sampler, config.batch_size, training_rng)
        batch = distillation_batch(
            train_dataset,
            indices;
            actuator_sensor_indices = actuators_to_sensors,
            window_size,
        )
        masked_observations = batch.observations .* reshape(input_mask, :, 1, 1)
        batch_loss = Ref(NaN)
        encoder_gradient, decoder_gradient = Flux.gradient(model.encoder, model.decoder) do trial_encoder, trial_decoder
            observation_representation, _ = trial_encoder(masked_observations)
            action_dimension = size(trial_decoder.embedding.weight, 2)
            shifted_actions = cat(
                zeros(Float32, action_dimension, 1, length(indices)),
                batch.expert_actions[:, 1:end-1, :];
                dims = 2,
            )
            predicted_actions, _ = trial_decoder(shifted_actions, observation_representation)
            loss = mean(abs2, predicted_actions .- batch.expert_actions)
            Zygote.ignore() do
                batch_loss[] = Float64(loss)
            end
            return loss
        end
        if !isfinite(batch_loss[])
            stop_on_numerical_failure!(
                archive_manager,
                update,
                "Non-finite training loss at update $update.";
                model,
                sampler,
                training_rng,
                losses,
                input_mask,
                operator_weights,
            )
        end
        Flux.update!(model.encoder_state_tree, model.encoder, encoder_gradient)
        Flux.update!(model.decoder_state_tree, model.decoder, decoder_gradient)

        if update <= config.regularized_updates && mod(update, config.proximal_interval) == 0
            if method_config.regularizer === :grouped
                apply_grouped_regularizer!(
                    model.encoder.embedding.weight;
                    groups,
                    regularization_strength,
                    theta_mode = method_config.theta_mode,
                )
            elseif method_config.regularizer === :group_reweighted
                apply_group_reweighted_regularizer!(
                    model.encoder.embedding.weight;
                    groups,
                    operator_weights,
                    regularization_strength,
                )
            else
                error("Unsupported regularizer $(method_config.regularizer).")
            end
        end
        if method_config.uses_operator_weights &&
           update <= config.regularized_updates &&
           mod(update, config.reweight_interval) == 0
            update_operator_weights!(operator_weights, model, groups)
        end

        if !model_is_finite(model)
            stop_on_numerical_failure!(
                archive_manager,
                update,
                "Non-finite apprentice parameters at update $update.";
                model,
                sampler,
                training_rng,
                losses,
                input_mask,
                operator_weights,
            )
        end
        push!(losses, batch_loss[])

        if !isnothing(archive_manager) && should_evaluate_candidates(archive_manager.schedule, update)
            evaluate_candidate_checkpoint!(
                archive_manager,
                model,
                update,
                validation_dataset,
                threshold_specs;
                groups,
                method,
                regularization_strength,
                validation_batch_size = config.validation_batch_size,
                prediction_mode = config.validation_prediction_mode,
                diagnostic_teacher_forced = config.diagnostic_teacher_forced,
            )
        end
        if !isnothing(archive_manager) && should_save_resume(archive_manager.schedule, update)
            save_resume_checkpoint!(
                archive_manager,
                update,
                Dict{Symbol, Any}(
                    :model => deepcopy(model),
                    :sampler => deepcopy(sampler),
                    :rng => deepcopy(training_rng),
                    :losses => copy(losses),
                    :finetune_input_mask => copy(input_mask),
                    :operator_weights => copy(operator_weights),
                ),
            )
        end
    end

    if !isnothing(archive_manager)
        if should_evaluate_candidates(archive_manager.schedule, total_updates; final = true)
            evaluate_candidate_checkpoint!(
                archive_manager,
                model,
                total_updates,
                validation_dataset,
                threshold_specs;
                groups,
                method,
                regularization_strength,
                validation_batch_size = config.validation_batch_size,
                prediction_mode = config.validation_prediction_mode,
                diagnostic_teacher_forced = config.diagnostic_teacher_forced,
            )
        end
        save_resume_checkpoint!(
            archive_manager,
            total_updates,
            Dict{Symbol, Any}(
                :model => deepcopy(model),
                :sampler => deepcopy(sampler),
                :rng => deepcopy(training_rng),
                :losses => copy(losses),
                :finetune_input_mask => copy(input_mask),
                :operator_weights => copy(operator_weights),
            );
            status = :complete,
        )
        finalize_pareto_archive!(archive_manager)
    end

    return (
        model,
        losses,
        final_input_mask = input_mask,
        groups,
        regularization_strength,
        archive_manager,
    )
end


function render_run_apprentice(input_mask; steps::Int = 200)
    length(input_mask) == size(env.state, 1) || throw(DimensionMismatch("Input mask has the wrong length."))
    rewards = Float64[]
    collected_actions = zeros(Float32, steps, actuators)
    reset!(env)
    generate_random_init()

    for step in 1:steps
        action = prob(apprentice, env.state .* input_mask, nothing).μ[:, :, 1]
        collected_actions[step, :] .= vec(action)
        env(action)
        push!(rewards, state_Nu(env))
    end
    return (reward_sum = sum(rewards), rewards, collected_actions)
end


# Apprentice outputs are protocol-specific and never overwrite shared
# infrastructure or repository-level ignore rules.
dirpath = get(
    ENV,
    "DISTILLATION_OUTPUT_DIRECTORY",
    joinpath(@__DIR__, "outputs", string(EXPERT_APPRENTICE_PROTOCOL)),
)
mkpath(dirpath)

function apprentice_save_stem(; group_channels_value = group_channels)
    method_tag = string(normalize_apprentice_kind(apprentice_training_kind))
    protocol_tag = string(EXPERT_APPRENTICE_PROTOCOL)
    grouping_tag = group_channels_value ? "grouped_channels" : "separate_channels"
    return "MAT_Apprentice_$(method_tag)_$(protocol_tag)_$(grouping_tag)"
end

function apprentice_save_path(number = nothing; group_channels_value = group_channels)
    stem = apprentice_save_stem(; group_channels_value)
    filename = isnothing(number) ? "$(stem).jld2" : "$(stem)_$(number).jld2"
    return joinpath(dirpath, "saves", filename)
end

function save_apprentice(
    input_mask;
    number = nothing,
    group_channels_value = group_channels,
    metadata = Dict{Symbol, Any}(),
)
    length(input_mask) == size(env.state, 1) || throw(DimensionMismatch("Input mask has the wrong length."))
    path = apprentice_save_path(number; group_channels_value)
    mkpath(dirname(path))
    pareto_atomic_save(
        path;
        apprentice,
        input_mask = Float32.(input_mask),
        method = normalize_apprentice_kind(apprentice_training_kind),
        protocol = EXPERT_APPRENTICE_PROTOCOL,
        group_channels = group_channels_value,
        metadata = Dict{Symbol, Any}(metadata),
    )
    return path
end

function load_apprentice(
    number = nothing;
    group_channels_value = group_channels,
)
    path = apprentice_save_path(number; group_channels_value)
    loaded = JLD2.load(path)
    loaded_method = Symbol(loaded["method"])
    normalize_apprentice_kind(loaded_method)
    loaded_protocol = Symbol(loaded["protocol"])
    loaded_protocol === EXPERT_APPRENTICE_PROTOCOL || error(
        "Saved apprentice protocol $loaded_protocol does not match $EXPERT_APPRENTICE_PROTOCOL.",
    )
    global apprentice = loaded["apprentice"]
    global apprentice_training_kind = loaded_method
    global apprentice_training_rIC = loaded_protocol === :varying
    return (
        apprentice,
        input_mask = Float32.(loaded["input_mask"]),
        metadata = get(loaded, "metadata", Dict{Symbol, Any}()),
        path,
    )
end

function load_apprentice_kind(
    kind::Symbol,
    number = nothing;
    group_channels_value = group_channels,
)
    global apprentice_training_kind = normalize_apprentice_kind(kind)
    return load_apprentice(number; group_channels_value)
end

go_load(number = nothing; kwargs...) =
    load_apprentice_kind(:go, number; kwargs...)
gr_load(number = nothing; kwargs...) =
    load_apprentice_kind(:gr, number; kwargs...)
group_lasso_load(number = nothing; kwargs...) =
    load_apprentice_kind(:group_lasso, number; kwargs...)
growl_load(number = nothing; kwargs...) =
    load_apprentice_kind(:growl, number; kwargs...)
