module MaskedStudy

using JLD2, Dates
include(joinpath(@__DIR__, "..", "Noise_Study", "NoiseStudy.jl"))
using .NoiseStudy: fingerprint, file_sha256, atomic_save, selected_candidate_record

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_RESULTS = joinpath(@__DIR__, "results")
const DEFAULT_COMPARISON = joinpath(ROOT, "Revision", "MAT_IPPO_Comparison", "results")
const BUDGETS = Dict(:fixed => 2000, :varying => 4000)

function select_candidate(protocol, grouping, root)
    protocol in (:fixed, :varying) || error("Invalid protocol")
    grouping in (:gc, :sc) || error("Invalid grouping")
    expected = protocol === :fixed ? :package7_fixed_regularizer_comparison : :package8_varying_regularizer_comparison
    records = map(("go", "gr", "group-lasso", "growl")) do method
        configuration = "$method-$grouping"
        path = joinpath(root, configuration, "analysis", "selected_test_candidate.jld2")
        record = selected_candidate_record(path)
        Symbol(JLD2.load(path, "experiment")) === expected || error("Wrong source experiment: $path")
        string(record.candidate[:configuration]) == configuration || error("Configuration mismatch")
        record
    end
    chosen = first(sort(collect(records); by = r -> (
        Int(r.candidate[:active_inputs]), Float64(r.candidate[:validation_matching]),
        string(r.candidate[:configuration]), string(r.candidate[:candidate_id]),
    )))
    mask = Float32.(chosen.candidate[:mask])
    length(mask) == 360 && all(x -> x in (0f0, 1f0), mask) && any(!iszero, mask) || error("Invalid input mask")
    isfinite(chosen.candidate[:validation_matching]) || error("Invalid validation MSE")
    audit = [(configuration = r.candidate[:configuration], active_inputs = r.candidate[:active_inputs],
              active_groups = r.candidate[:active_groups], validation_mse = r.candidate[:validation_matching],
              candidate_id = r.candidate[:candidate_id]) for r in records]
    return (; protocol, grouping, mask, configuration = chosen.candidate[:configuration],
        candidate_id = chosen.candidate[:candidate_id], active_groups = chosen.candidate[:active_groups],
        active_inputs = chosen.candidate[:active_inputs], validation_mse = chosen.candidate[:validation_matching],
        selection_sha256 = chosen.selection_sha256, checkpoint_sha256 = chosen.checkpoint_sha256,
        selection_path = chosen.selection_path, checkpoint_path = chosen.checkpoint_path, audit)
end

function baseline_record(comparison, entry, protocol)
    path = joinpath(comparison, "runs", entry.run_id, string(protocol), "mat.jld2")
    JLD2.jldopen(path, "r") do f
        f["status"] == "complete" || error("Incomplete dense reference: $path")
        Symbol(f["protocol"]) === protocol || error("Dense protocol mismatch")
        f["run_seed"] == entry.run_seed && f["ic_seed"] == entry.ic_seed || error("Dense seed mismatch")
        f["episode_target"] == f["episodes_completed"] == BUDGETS[protocol] || error("Dense budget mismatch")
        Symbol(f["config_name"]) === :modified_full || error("Dense MAT configuration mismatch")
        protocol === :varying && f["initial_condition_trace"] != entry.varying_trace && error("Dense IC trace mismatch")
        parameters = f["run_parameters"]
        get(parameters, "Ra", 1e4) == 1e4 || error("Dense reference is not Ra=1e4")
        initial_hash = haskey(f, "initial_parameter_hash") ? f["initial_parameter_hash"] : nothing
        legacy_initial_hash = haskey(f, "full_initial_hash") ? f["full_initial_hash"] : nothing
        return (; path = abspath(path), sha256 = file_sha256(path), parameters, initial_hash, legacy_initial_hash)
    end
end

function build_manifest(; comparison = DEFAULT_COMPARISON,
    package7 = joinpath(ROOT, "Revision", "Package7", "results", "260830_173924"),
    package8 = joinpath(ROOT, "Revision", "Package8", "results", "260830_231109"))
    source_plan = joinpath(comparison, "run_plan.jld2")
    plan = JLD2.load(source_plan, "plan")
    plan["schema_version"] == 1 || error("Unsupported comparison plan")
    entries = plan["entries"]
    length(entries) == 10 || error("Expected exactly ten Comparison seed pairs; found $(length(entries)).")
    length(unique(e.run_id for e in entries)) == 10 || error("Duplicate run IDs")
    length(unique((e.run_seed, e.ic_seed) for e in entries)) == 10 || error("Duplicate seed pairs")
    all(length(e.varying_trace) == 4000 && all(c.split === :train for c in e.varying_trace) for e in entries) || error("Invalid training IC schedules")
    candidates = [select_candidate(p, g, p === :fixed ? package7 : package8) for p in (:fixed, :varying) for g in (:gc, :sc)]
    baselines = Dict((e.run_id, p) => baseline_record(comparison, e, p) for e in entries for p in (:fixed, :varying))
    sources = [joinpath(@__DIR__, name) for name in ("MaskedStudy.jl", "MaskedTraining.jl", "run_worker.jl")]
    append!(sources, [joinpath(ROOT, "Revision", "Run_Files", "$(p)IC_MAT.jl") for p in ("Fixed", "Varying")])
    push!(sources, joinpath(ROOT, "Revision", "MAT_IPPO_Comparison", "MATIPPOExperiment.jl"))
    corpus = joinpath(ROOT, "Revision", "VaryingIC_Corpus", "varying_ic_corpus.jld2")
    source_hashes = Dict(relpath(p, ROOT) => file_sha256(p) for p in sources)
    source_hashes[relpath(corpus, ROOT)] = file_sha256(corpus)
    rl_sources = Dict(name => file_sha256(joinpath(dirname(Base.find_package("RL")), name))
        for name in ("agent_mat.jl", "agent.jl", "env.jl", "hook.jl"))
    payload = (; schema_version = 1, rayleigh = 1e4, entries, candidates, baselines,
        comparison_plan_sha256 = file_sha256(source_plan), source_hashes, rl_sources,
        selection_rule = "minimum active inputs, then minimum validation MSE; frozen Package-7/8 candidates only")
    # Arrays (including masks) are explicitly flattened in the identity payload.
    return (; payload..., identity = fingerprint(payload))
end

manifest_path(results) = joinpath(results, "manifest.jld2")
result_path(results, protocol, grouping, run_id) = joinpath(results, "runs", string(protocol), string(grouping), "$run_id.jld2")

function complete_result(path, identity, budget)
    isfile(path) || return false
    JLD2.jldopen(path, "r") do f
        f["manifest_identity"] == identity || error("Result belongs to a different manifest: $path")
        f["episode_target"] == budget || error("Result budget mismatch: $path")
        f["status"] == "complete" && f["episodes_completed"] == budget
    end
end

function parse_options(args, defaults; flags = String[])
    options = copy(defaults)
    i = 1
    while i <= length(args)
        key = replace(args[i], r"^--" => "", "-" => "_")
        haskey(options, key) || error("Unknown option $(args[i])")
        if key in flags
            options[key] = true
        else
            i += 1
            i <= length(args) || error("Missing option value")
            options[key] = args[i]
        end
        i += 1
    end
    options
end
end
