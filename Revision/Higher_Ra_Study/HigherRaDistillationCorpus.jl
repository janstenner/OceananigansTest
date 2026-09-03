ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"

include(joinpath(@__DIR__, "..", "Expert_Apprentice_Distillation", "DistillationCorpus.jl"))

const HIGHER_RA_STUDIES = (
    ra5e4 = (
        tag = :ra5e4,
        label = "Ra=5e4",
        rayleigh = 5.0e4,
        run_file = normpath(joinpath(@__DIR__, "..", "Run_Files", "VaryingIC_MAT_Ra5e4.jl")),
        state_corpus = normpath(joinpath(@__DIR__, "..", "VaryingIC_Corpus", "varying_ic_corpus_Ra5e4.jld2")),
        expert = joinpath(@__DIR__, "experts", "ra5e4", "expert.jld2"),
    ),
    ra1e5 = (
        tag = :ra1e5,
        label = "Ra=1e5",
        rayleigh = 1.0e5,
        run_file = normpath(joinpath(@__DIR__, "..", "Run_Files", "VaryingIC_MAT_Ra1e5.jl")),
        state_corpus = normpath(joinpath(@__DIR__, "..", "VaryingIC_Corpus", "varying_ic_corpus_Ra1e5.jld2")),
        expert = joinpath(@__DIR__, "experts", "ra1e5", "expert.jld2"),
    ),
)

function normalize_higher_ra_study(value)::Symbol
    tag = Symbol(lowercase(replace(strip(string(value)), "-" => "")))
    tag = get(Dict(:ra50000 => :ra5e4, :ra100000 => :ra1e5), tag, tag)
    tag in keys(HIGHER_RA_STUDIES) || throw(ArgumentError(
        "Higher-Ra study must be ra5e4 or ra1e5, got '$value'.",
    ))
    return tag
end

higher_ra_study(value) = getproperty(HIGHER_RA_STUDIES, normalize_higher_ra_study(value))

higher_ra_worker_root(study) = joinpath(
    @__DIR__,
    "Distillation_Corpuses",
    string(normalize_higher_ra_study(study)),
    "worker_results",
)

function higher_ra_dict_value(mapping, key::Symbol)
    haskey(mapping, key) && return mapping[key]
    haskey(mapping, string(key)) && return mapping[string(key)]
    error("Missing '$key' in Higher-Ra data.")
end

function validate_higher_ra_sources(study; expert_path = study.expert)
    isfile(study.run_file) || error("Higher-Ra MAT run file is missing: $(study.run_file)")
    isfile(study.state_corpus) || error("Higher-Ra state corpus is missing: $(study.state_corpus)")
    isfile(expert_path) || error("Higher-Ra expert is missing: $expert_path")
    validate_agent_only = isdefined(Main, :MATExpertTraining) &&
        isdefined(Main.MATExpertTraining, :validate_agent_only_checkpoint)
    validate_agent_only && Main.MATExpertTraining.validate_agent_only_checkpoint(expert_path)
    return (
        expert_path = abspath(expert_path),
        expert_sha256 = file_sha256(expert_path),
        run_file_path = abspath(study.run_file),
        run_file_sha256 = file_sha256(study.run_file),
        state_corpus_path = abspath(study.state_corpus),
        state_corpus_sha256 = file_sha256(study.state_corpus),
    )
end

function load_higher_ra_corpus_plan(study)
    loaded = JLD2.load(study.state_corpus)
    haskey(loaded, "corpus") || error("State corpus has no 'corpus' entry: $(study.state_corpus)")
    haskey(loaded, "simulation_config") || error(
        "State corpus has no 'simulation_config' entry: $(study.state_corpus)",
    )
    config = loaded["simulation_config"]
    observed_ra = Float64(higher_ra_dict_value(config, :Ra))
    observed_ra == study.rayleigh || error(
        "$(study.label) corpus reports Rayleigh number $observed_ra.",
    )
    corpus = loaded["corpus"]
    seeds = Dict{Symbol, Vector{Int}}()
    for split in DISTILLATION_SPLITS
        records = higher_ra_dict_value(corpus, split)
        seeds[split] = sort!(Int.(collect(keys(records))))
        expected = DISTILLATION_EXPECTED_BASES[split]
        length(seeds[split]) == expected || error(
            "Expected $expected $(study.label) $split bases, found $(length(seeds[split])).",
        )
    end
    return seeds
end

function higher_ra_worker_matches(
    path,
    study,
    split,
    expected_expert_sha,
    expected_corpus_sha,
    expected_run_file_sha,
)
    isfile(path) || return false
    return try
        result = load_distillation_worker(path)
        expert_metadata = distillation_value(result, :expert_metadata)
        Int(distillation_value(result, :rollout_steps)) == DISTILLATION_ROLLOUT_STEPS &&
        collect(distillation_value(result, :offsets)) == distillation_offsets(:varying, split) &&
        string(higher_ra_dict_value(expert_metadata, :higher_ra_study)) == string(study.tag) &&
        Float64(higher_ra_dict_value(expert_metadata, :rayleigh)) == study.rayleigh &&
        string(higher_ra_dict_value(expert_metadata, :checkpoint_sha256)) == expected_expert_sha &&
        string(higher_ra_dict_value(expert_metadata, :state_corpus_sha256)) == expected_corpus_sha &&
        string(higher_ra_dict_value(expert_metadata, :run_file_sha256)) == expected_run_file_sha
    catch
        false
    end
end

