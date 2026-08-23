include(joinpath(@__DIR__, "MATExpertTraining.jl"))
using .MATExpertTraining
using Dates
using Flux
using JLD2
using UUIDs

const RESULTS_DIRECTORY = MATExpertTraining.DEFAULT_RESULTS_DIRECTORY
const DISTILLATION_DIRECTORY = MATExpertTraining.DEFAULT_DISTILLATION_EXPERT_DIRECTORY

function rank_one_candidate(protocol)
    return only(
        candidate for candidate in MATExpertTraining.CANDIDATES
        if candidate.protocol === protocol && candidate.rank == 1
    )
end

function policy_state_equal(first_agent, second_agent)
    return isequal(Flux.state(first_agent.policy), Flux.state(second_agent.policy))
end

function validate_threshold_varying(results_directory, compact_expert_path)
    manifest_path = joinpath(results_directory, "varying", "candidate.jld2")
    checkpoint_path = joinpath(results_directory, "varying", "best_so_far.jld2")
    isfile(manifest_path) || error("Missing Varying candidate manifest: $manifest_path")
    isfile(checkpoint_path) || error("Missing Varying threshold checkpoint: $checkpoint_path")
    manifest = JLD2.load(manifest_path)
    string(manifest["selection_mode"]) == "threshold" || error(
        "Varying candidate was not selected by threshold.",
    )
    Bool(manifest["threshold_reached"]) || error("Varying threshold was not reached.")
    Int(manifest["additional_episodes"]) == 703 || error(
        "Expected 703 additional Varying episodes.",
    )
    Float64(manifest["criterion_value"]) > MATExpertTraining.VARYING_THRESHOLD || error(
        "Varying candidate does not exceed the configured threshold.",
    )
    checkpoint = JLD2.jldopen(checkpoint_path, "r") do file
        (
            status = string(read(file, "status")),
            protocol = Symbol(read(file, "protocol")),
            run_id = string(read(file, "run_id")),
            additional_episodes = Int(read(file, "additional_episodes")),
            total_episodes = Int(read(file, "total_episodes")),
            criterion_value = Float64(read(file, "criterion_value")),
            threshold_reached = Bool(read(file, "threshold_reached")),
        )
    end
    checkpoint.status == "best_so_far" || error("Unexpected Varying checkpoint status.")
    checkpoint.protocol === :varying || error("Varying checkpoint protocol mismatch.")
    checkpoint.run_id == string(manifest["winner_run_id"]) || error(
        "Varying checkpoint run-ID mismatch.",
    )
    checkpoint.additional_episodes == Int(manifest["additional_episodes"]) || error(
        "Varying checkpoint episode mismatch.",
    )
    checkpoint.total_episodes == Int(manifest["total_episodes"]) || error(
        "Varying checkpoint total-episode mismatch.",
    )
    checkpoint.threshold_reached || error("Varying checkpoint did not reach threshold.")
    isapprox(
        checkpoint.criterion_value,
        Float64(manifest["criterion_value"]);
        atol = 1e-10,
        rtol = 0.0,
    ) || error("Varying checkpoint criterion mismatch.")

    MATExpertTraining.validate_agent_only_checkpoint(compact_expert_path)
    checkpoint_agent = JLD2.load(checkpoint_path, "agent")
    compact_agent = JLD2.load(compact_expert_path, "agent")
    policy_state_equal(checkpoint_agent, compact_agent) || error(
        "The compact Varying expert policy differs from the threshold checkpoint.",
    )
    return (;
        manifest_path = abspath(manifest_path),
        checkpoint_path = abspath(checkpoint_path),
        run_id = checkpoint.run_id,
        checkpoint_sha256 = MATExpertTraining.source_hash(checkpoint_path),
        additional_episodes = checkpoint.additional_episodes,
        total_episodes = checkpoint.total_episodes,
        criterion_value = checkpoint.criterion_value,
    )
end

function stage_expert(source_path, target_path)
    MATExpertTraining.validate_agent_only_checkpoint(source_path)
    mkpath(dirname(target_path))
    temporary_path = joinpath(
        dirname(target_path),
        ".$(basename(target_path)).$(getpid()).$(time_ns()).$(uuid4()).tmp",
    )
    cp(source_path, temporary_path; force = true)
    MATExpertTraining.validate_agent_only_checkpoint(temporary_path)
    source_sha256 = MATExpertTraining.source_hash(source_path)
    MATExpertTraining.source_hash(temporary_path) == source_sha256 || error(
        "Staged expert hash differs from source: $source_path",
    )
    return (; source_path = abspath(source_path), target_path = abspath(target_path),
            temporary_path, source_sha256)
end

function main()
    fixed_candidate = rank_one_candidate(:fixed)
    fixed_source = MATExpertTraining.verified_candidate_record(
        fixed_candidate,
        MATExpertTraining.DEFAULT_SOURCE_RESULTS_DIRECTORY,
    )
    fixed_compact_path = MATExpertTraining.best_so_far_expert_path(
        RESULTS_DIRECTORY,
        :fixed,
    )
    MATExpertTraining.save_compact_expert_from_checkpoint!(
        fixed_source.checkpoint_path,
        fixed_compact_path,
    )
    fixed_source_agent = JLD2.load(fixed_source.checkpoint_path, "agent")
    fixed_compact_agent = JLD2.load(fixed_compact_path, "agent")
    policy_state_equal(fixed_source_agent, fixed_compact_agent) || error(
        "The compact Fixed expert policy differs from the Package-4 checkpoint.",
    )

    varying_compact_path = MATExpertTraining.best_so_far_expert_path(
        RESULTS_DIRECTORY,
        :varying,
    )
    varying = validate_threshold_varying(RESULTS_DIRECTORY, varying_compact_path)

    selected = (
        (
            protocol = :fixed,
            run_id = fixed_candidate.run_id,
            selection = :package4_validation_rank_1,
            source_checkpoint_path = fixed_source.checkpoint_path,
            source_checkpoint_sha256 = fixed_source.checkpoint_sha256,
            additional_episodes = 0,
            criterion_value = -545.8045628433928,
            compact_expert_path = abspath(fixed_compact_path),
        ),
        (
            protocol = :varying,
            run_id = varying.run_id,
            selection = :rolling_100_threshold,
            source_checkpoint_path = varying.checkpoint_path,
            source_checkpoint_sha256 = varying.checkpoint_sha256,
            additional_episodes = varying.additional_episodes,
            criterion_value = varying.criterion_value,
            compact_expert_path = abspath(varying_compact_path),
        ),
    )

    staged = NamedTuple[]
    try
        for record in selected
            target_path = MATExpertTraining.distillation_expert_path(
                DISTILLATION_DIRECTORY,
                record.protocol,
            )
            push!(staged, stage_expert(record.compact_expert_path, target_path))
        end
        for record in staged
            mv(record.temporary_path, record.target_path; force = true)
        end
    finally
        for record in staged
            isfile(record.temporary_path) && rm(record.temporary_path; force = true)
        end
    end

    publication_records = map(zip(selected, staged)) do (selection, publication)
        MATExpertTraining.validate_agent_only_checkpoint(publication.target_path)
        MATExpertTraining.source_hash(publication.target_path) == publication.source_sha256 ||
            error("Published expert hash mismatch: $(publication.target_path)")
        merge(selection, (
            compact_expert_sha256 = publication.source_sha256,
            distillation_path = publication.target_path,
        ))
    end
    manifest_path = joinpath(RESULTS_DIRECTORY, "expert_publication_manifest.jld2")
    MATExpertTraining.atomic_jldsave(
        manifest_path;
        status = "complete",
        publication = :fixed_package4_rank1_and_varying_threshold_winner,
        published_at = string(now()),
        experts = collect(publication_records),
    )
    println("Published selected Fixed and Varying Distillation experts:")
    for record in publication_records
        println(
            "  $(record.protocol): $(record.run_id), additional_episodes=" *
            "$(record.additional_episodes), sha256=$(record.compact_expert_sha256)",
        )
        println("    $(record.distillation_path)")
    end
    println("Publication manifest: $(abspath(manifest_path))")
    return manifest_path
end

main()
