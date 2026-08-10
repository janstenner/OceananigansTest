module Package6Analysis

using Statistics

export valueof, normalize_record, scientific_front, front_envelope, front_regret,
       empirical_attainment, hitting_metrics, checkpoint_metrics, late_metrics,
       reset_metrics, archive_convergence, spearman_correlation, jaccard,
       mask_stability, select_test_candidates, excursion_summary, downsample_indices

valueof(container, key::Symbol; default = missing) = haskey(container, key) ? container[key] :
    haskey(container, string(key)) ? container[string(key)] :
    default !== missing ? default : error("Missing '$key'.")

function normalize_record(raw; metadata = NamedTuple())
    record = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in raw)
    for (key, value) in pairs(metadata)
        record[key] = value
    end
    record[:update] = Int(valueof(record, :update))
    record[:active_groups] = Int(valueof(record, :active_groups))
    record[:validation_matching] = Float64(valueof(record, :validation_matching))
    record[:numeric_status] = Symbol(valueof(record, :numeric_status; default = :ok))
    return record
end

valid_record(record) = record[:numeric_status] === :ok &&
    isfinite(record[:validation_matching]) && record[:validation_matching] >= 0 &&
    0 <= record[:active_groups] <= 96

record_key(record) = (
    record[:active_groups],
    record[:validation_matching],
    record[:update],
    string(valueof(record, :run_id; default = "")),
    string(valueof(record, :candidate_id; default = "")),
)

"""Nondominated front on active SC groups (smaller) and validation MSE (smaller)."""
function scientific_front(records)
    ordered = sort([record for record in records if valid_record(record)]; by = record_key)
    front = Dict{Symbol, Any}[]
    best_mse = Inf
    seen_objectives = Set{Tuple{Int, Float64}}()
    for record in ordered
        objectives = (record[:active_groups], record[:validation_matching])
        objectives in seen_objectives && continue
        if record[:validation_matching] < best_mse
            push!(front, record)
            push!(seen_objectives, objectives)
            best_mse = record[:validation_matching]
        end
    end
    return front
end

"""Best front MSE at the same or a sparser (smaller-group) point."""
function front_envelope(front, active_groups::Integer)
    values = Float64[
        record[:validation_matching] for record in front
        if record[:active_groups] <= active_groups && valid_record(record)
    ]
    return isempty(values) ? Inf : minimum(values)
end

function front_regret(record, front)
    baseline = front_envelope(front, record[:active_groups])
    isfinite(baseline) || return Inf
    baseline == 0 && return record[:validation_matching] == 0 ? 0.0 : Inf
    return max(0.0, record[:validation_matching] / baseline - 1.0)
end

function empirical_attainment(run_fronts::AbstractDict; group_counts = 0:96)
    replicate_ids = sort!(collect(keys(run_fronts)); by = string)
    isempty(replicate_ids) && return NamedTuple[]
    rows = NamedTuple[]
    for groups in group_counts
        envelopes = sort(Float64[front_envelope(run_fronts[id], groups) for id in replicate_ids])
        for required in 1:length(replicate_ids)
            push!(rows, (
                active_groups = Int(groups),
                attained_seeds = required,
                seed_count = length(replicate_ids),
                attainment_fraction = required / length(replicate_ids),
                validation_mse = envelopes[required],
            ))
        end
    end
    return rows
end

function hitting_metrics(records; targets = 0:96)
    ordered = sort([record for record in records if valid_record(record)]; by = record -> record[:update])
    return [begin
        index = findfirst(record -> record[:active_groups] <= target, ordered)
        isnothing(index) ? (
            target_groups = Int(target), reachable = false, first_update = missing,
            active_groups_at_hit = missing, validation_mse_at_hit = missing,
        ) : (
            target_groups = Int(target), reachable = true, first_update = ordered[index][:update],
            active_groups_at_hit = ordered[index][:active_groups],
            validation_mse_at_hit = ordered[index][:validation_matching],
        )
    end for target in targets]
end

function checkpoint_metrics(records, own_front, strength_front)
    return [begin
        strength_envelope = front_envelope(strength_front, record[:active_groups])
        merge(record, Dict{Symbol, Any}(
            :own_front_regret => front_regret(record, own_front),
            :strength_front_regret => front_regret(record, strength_front),
            :front_near => isfinite(strength_envelope) && record[:validation_matching] <= 1.10 * strength_envelope,
        ))
    end for record in sort(records; by = record -> record[:update]) if valid_record(record)]
end

function excursion_summary(metrics)
    isempty(metrics) && return (
        excursion_count = 0, recovery_updates = Int[], unresolved_end_excursions = 0,
    )
    count = 0
    recovery = Int[]
    unresolved = 0
    index = 1
    while index <= length(metrics)
        if metrics[index][:front_near]
            index += 1
            continue
        end
        count += 1
        start_update = metrics[index][:update]
        while index <= length(metrics) && !metrics[index][:front_near]
            index += 1
        end
        if index <= length(metrics)
            push!(recovery, metrics[index][:update] - start_update)
        else
            unresolved += 1
        end
    end
    return (excursion_count = count, recovery_updates = recovery, unresolved_end_excursions = unresolved)
end

function finite_quantile(values, probability)
    finite_values = Float64[value for value in values if isfinite(value)]
    return isempty(finite_values) ? NaN : quantile(finite_values, probability)
end

function late_metrics(metrics, total_updates::Integer; windows = (0.10, 0.20, 0.30))
    rows = NamedTuple[]
    for fraction in windows
        start_update = ceil(Int, (1 - fraction) * total_updates)
        selected = filter(record -> record[:update] >= start_update, metrics)
        excursion = excursion_summary(selected)
        push!(rows, (
            window_fraction = Float64(fraction),
            start_update,
            checkpoint_count = length(selected),
            front_near_fraction = isempty(selected) ? NaN : mean(record[:front_near] for record in selected),
            median_own_front_regret = finite_quantile((record[:own_front_regret] for record in selected), 0.5),
            p90_own_front_regret = finite_quantile((record[:own_front_regret] for record in selected), 0.9),
            median_strength_front_regret = finite_quantile((record[:strength_front_regret] for record in selected), 0.5),
            p90_strength_front_regret = finite_quantile((record[:strength_front_regret] for record in selected), 0.9),
            excursion_count = excursion.excursion_count,
            median_recovery_updates = finite_quantile(excursion.recovery_updates, 0.5),
            p90_recovery_updates = finite_quantile(excursion.recovery_updates, 0.9),
            unresolved_end_excursions = excursion.unresolved_end_excursions,
        ))
    end
    return rows
end

function reset_metrics(records, total_updates::Integer)
    ordered = sort([record for record in records if valid_record(record)]; by = record -> record[:update])
    events = NamedTuple[]
    for index in 2:length(ordered)
        previous, current = ordered[index - 1], ordered[index]
        group_delta = current[:active_groups] - previous[:active_groups]
        mse_ratio = previous[:validation_matching] == 0 ? Inf : current[:validation_matching] / previous[:validation_matching]
        group_reset = group_delta > 0
        mse_reset = mse_ratio > 1.10
        (group_reset || mse_reset) && push!(events, (
            update = current[:update],
            group_reset,
            mse_reset,
            joint_reset = group_reset && mse_reset,
            group_delta,
            mse_ratio,
            mse_relative_jump = mse_ratio - 1,
        ))
    end
    scale = total_updates == 0 ? NaN : 1000 / total_updates
    summary = (
        group_reset_count = count(event -> event.group_reset, events),
        mse_reset_count = count(event -> event.mse_reset, events),
        joint_reset_count = count(event -> event.joint_reset, events),
        group_reset_rate_per_1000 = count(event -> event.group_reset, events) * scale,
        mse_reset_rate_per_1000 = count(event -> event.mse_reset, events) * scale,
        joint_reset_rate_per_1000 = count(event -> event.joint_reset, events) * scale,
        median_group_jump = finite_quantile((event.group_delta for event in events if event.group_reset), 0.5),
        median_mse_jump = finite_quantile((event.mse_relative_jump for event in events if event.mse_reset), 0.5),
    )
    return (; events, summary)
end

function archive_convergence(records, final_front)
    ordered = sort([record for record in records if valid_record(record)]; by = record -> record[:update])
    group_caps = sort!(unique(record[:active_groups] for record in final_front))
    rows = NamedTuple[]
    for update in sort!(unique(record[:update] for record in ordered))
        history = filter(record -> record[:update] <= update, ordered)
        covered = count(group_caps) do groups
            final_value = front_envelope(final_front, groups)
            current_value = front_envelope(history, groups)
            isfinite(final_value) && current_value <= 1.10 * final_value
        end
        push!(rows, (
            update,
            covered_group_caps = covered,
            final_group_cap_count = length(group_caps),
            coverage = isempty(group_caps) ? NaN : covered / length(group_caps),
        ))
    end
    first_at(threshold) = begin
        index = findfirst(row -> row.coverage >= threshold, rows)
        isnothing(index) ? missing : rows[index].update
    end
    return (rows, updates_to_90 = first_at(0.9), updates_to_100 = first_at(1.0))
end

function tied_ranks(values)
    order = sortperm(values)
    ranks = zeros(Float64, length(values))
    index = 1
    while index <= length(order)
        stop = index
        while stop < length(order) && values[order[stop + 1]] == values[order[index]]
            stop += 1
        end
        rank = (index + stop) / 2
        ranks[order[index:stop]] .= rank
        index = stop + 1
    end
    return ranks
end

function spearman_correlation(left, right)
    length(left) == length(right) || throw(DimensionMismatch("Spearman inputs must have equal length."))
    length(left) >= 2 || return NaN
    x, y = tied_ranks(collect(left)), tied_ranks(collect(right))
    return (std(x) == 0 || std(y) == 0) ? NaN : cor(x, y)
end

function jaccard(left, right)
    a, b = BitVector(vec(left)), BitVector(vec(right))
    length(a) == length(b) || throw(DimensionMismatch("Masks must have equal length."))
    union_count = count(a .| b)
    return union_count == 0 ? 1.0 : count(a .& b) / union_count
end

function mask_stability(run_fronts::AbstractDict, all_run_records::AbstractDict; mse_threshold = 0.01)
    run_ids = sort!(collect(keys(run_fronts)); by = string)
    pair_rows = NamedTuple[]
    if length(run_ids) >= 2
        for left_index in 1:(length(run_ids) - 1), right_index in (left_index + 1):length(run_ids)
            left_id, right_id = run_ids[left_index], run_ids[right_index]
            left_by_group = Dict(record[:active_groups] => record for record in run_fronts[left_id])
            right_by_group = Dict(record[:active_groups] => record for record in run_fronts[right_id])
            for groups in sort!(collect(intersect(keys(left_by_group), keys(right_by_group))))
                push!(pair_rows, (
                    left_run = string(left_id), right_run = string(right_id), active_groups = groups,
                    jaccard = jaccard(left_by_group[groups][:global_mask], right_by_group[groups][:global_mask]),
                ))
            end
        end
    end
    selected = Dict{String, Dict{Symbol, Any}}()
    for run_id in run_ids
        qualified = filter(record -> valid_record(record) && record[:validation_matching] <= mse_threshold, all_run_records[run_id])
        isempty(qualified) && continue
        selected[string(run_id)] = first(sort(qualified; by = record_key))
    end
    frequency = isempty(selected) ? Float64[] : begin
        masks = [Float64.(vec(record[:global_mask])) for record in values(selected)]
        vec(mean(reduce(hcat, masks); dims = 2))
    end
    return (pair_rows, selected, selection_frequency = frequency)
end

function candidate_order(record)
    return (
        record[:validation_matching], record[:update], string(record[:run_id]),
    )
end

function sparse_order(record)
    return (
        record[:active_groups], record[:validation_matching], record[:update], string(record[:run_id]),
    )
end

function select_test_candidates(go_records; mse_threshold = 0.01)
    front = scientific_front(go_records)
    isempty(front) && error("The pooled GO front is empty.")
    match = first(sort(front; by = candidate_order))
    qualified = filter(record -> record[:validation_matching] <= mse_threshold, front)
    sparse = isempty(qualified) ? nothing : first(sort(qualified; by = sparse_order))
    if !isnothing(sparse) && string(sparse[:candidate_id]) == string(match[:candidate_id])
        sparse = nothing
    end
    return (; match, sparse, front)
end

function downsample_indices(records; maximum_points::Int = 1200, preserve = Int[])
    count_records = length(records)
    count_records <= maximum_points && return collect(1:count_records)
    stride = max(1, ceil(Int, count_records / maximum_points))
    selected = Set(1:stride:count_records)
    push!(selected, 1, count_records)
    union!(selected, preserve)
    return sort!(collect(filter(index -> 1 <= index <= count_records, selected)))
end

end
