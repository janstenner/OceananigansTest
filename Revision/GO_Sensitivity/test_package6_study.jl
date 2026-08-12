using Test

include(joinpath(@__DIR__, "Package6Study.jl"))
include(joinpath(@__DIR__, "Package6Analysis.jl"))
using .Package6Study
using .Package6Analysis

record(groups, mse, update; id = "c$update", run = "run", mask = BitVector([true, false, true])) = Dict{Symbol, Any}(
    :active_groups => groups,
    :validation_matching => mse,
    :update => update,
    :candidate_id => id,
    :run_id => run,
    :numeric_status => :ok,
    :global_mask => mask,
)

@testset "Package 6 manifest and paired seed plan" begin
    jobs = study_jobs()
    analyses = analysis_jobs()
    @test length(jobs) == 36
    @test count(job -> job.method === :go, jobs) == 30
    @test count(job -> job.method === :gr, jobs) == 6
    @test length(analyses) == 2
    @test length(unique(job.id for job in jobs)) == 36
    @test length(unique(job.relative_path for job in jobs)) == 36
    @test all(short_path_components, jobs)
    @test length(study_jobs(:fixed)) == 18
    @test length(analysis_jobs(:fixed)) == 1
    @test P6_TRAINING_BATCH_SIZE == Dict(:fixed => 50, :varying => 100)
    @test P6_VALIDATION_BATCH_SIZE == Dict(:fixed => 200, :varying => 512)
    @test Set(job.regularization_strength for job in jobs if job.method === :go) == Set(P6_STRENGTHS)
    for protocol in (:fixed, :varying), replicate in 1:3
        paired = filter(job -> job.protocol === protocol && job.replicate == replicate, jobs)
        @test length(paired) == 6
        @test length(unique(job.apprentice_seed for job in paired)) == 1
        @test length(unique(job.batch_seed for job in paired)) == 1
        @test length(unique(job.pairing_hash for job in paired)) == 1
    end
    @test length(unique(seed_plan(replicate).apprentice_seed for replicate in 1:3)) == 3
    @test length(unique(seed_plan(replicate).batch_seed for replicate in 1:3)) == 3
    @test all(seed_plan(replicate).apprentice_seed != 600_601 for replicate in 1:3)
    @test seed_plan(1) == (replicate = 1, apprentice_seed = 177_701_484, batch_seed = 1_685_563_253)
end

@testset "Scientific fronts, attainment and regret" begin
    records = [
        record(4, 0.4, 0),
        record(3, 0.5, 1),
        record(3, 0.3, 2),
        record(2, 0.8, 3),
        record(4, 0.6, 4),
    ]
    front = scientific_front(records)
    @test [(r[:active_groups], r[:validation_matching]) for r in front] == [(2, 0.8), (3, 0.3)]
    @test front_envelope(front, 2) == 0.8
    @test front_envelope(front, 4) == 0.3
    @test front_regret(record(4, 0.33, 9), front) ≈ 0.1
    @test checkpoint_metrics([record(4, 0.33, 9)], front, front)[1][:front_near]

    attainment = empirical_attainment(Dict(
        1 => scientific_front([record(2, 0.2, 1)]),
        2 => scientific_front([record(2, 0.3, 1)]),
        3 => scientific_front([record(2, 0.4, 1)]),
    ); group_counts = 2:2)
    @test [row.validation_mse for row in attainment] == [0.2, 0.3, 0.4]
    @test [row.attainment_fraction for row in attainment] == [1 / 3, 2 / 3, 1.0]
end

@testset "Hitting, excursions, resets and archive coverage" begin
    records = [record(5, 1.0, 0), record(3, 0.5, 25), record(4, 0.7, 50), record(2, 0.4, 75)]
    hits = hitting_metrics(records; targets = (4, 2, 1))
    @test hits[1].first_update == 25
    @test hits[2].first_update == 75
    @test !hits[3].reachable

    measured = [
        merge(record(5, 1.0, 0), Dict(:front_near => true, :own_front_regret => 0.0, :strength_front_regret => 0.0)),
        merge(record(5, 2.0, 25), Dict(:front_near => false, :own_front_regret => 1.0, :strength_front_regret => 1.0)),
        merge(record(4, 1.0, 50), Dict(:front_near => false, :own_front_regret => 0.5, :strength_front_regret => 0.5)),
        merge(record(3, 0.5, 75), Dict(:front_near => true, :own_front_regret => 0.0, :strength_front_regret => 0.0)),
        merge(record(3, 0.9, 100), Dict(:front_near => false, :own_front_regret => 0.8, :strength_front_regret => 0.8)),
    ]
    excursions = excursion_summary(measured)
    @test excursions.excursion_count == 2
    @test excursions.recovery_updates == [50]
    @test excursions.unresolved_end_excursions == 1

    resets = reset_metrics(records, 100)
    @test resets.summary.group_reset_count == 1
    @test resets.summary.mse_reset_count == 1
    @test resets.summary.joint_reset_count == 1
    convergence = archive_convergence(records, scientific_front(records))
    @test last(convergence.rows).coverage == 1.0
    @test convergence.updates_to_100 !== missing
end

@testset "Masks and strength trends" begin
    @test jaccard(BitVector([true, false, true]), BitVector([true, true, false])) ≈ 1 / 3
    @test jaccard(falses(3), falses(3)) == 1.0
    @test spearman_correlation([1, 2, 3], [3, 2, 1]) ≈ -1.0
    fronts = Dict(
        "a" => [record(2, 0.1, 1; run = "a", mask = BitVector([true, false, true]))],
        "b" => [record(2, 0.2, 1; run = "b", mask = BitVector([true, true, false]))],
    )
    stability = mask_stability(fronts, fronts; mse_threshold = 0.2)
    @test length(stability.pair_rows) == 1
    @test stability.pair_rows[1].jaccard ≈ 1 / 3
    @test stability.selection_frequency ≈ [1.0, 0.5, 0.5]
end

@testset "Deterministic validation-only candidate selection" begin
    candidates = [
        record(8, 0.004, 100; id = "match", run = "b"),
        record(2, 0.009, 200; id = "sparse", run = "a"),
        record(1, 0.02, 50; id = "too_sparse", run = "c"),
    ]
    selection = select_test_candidates(candidates)
    @test selection.match[:candidate_id] == "match"
    @test selection.sparse[:candidate_id] == "sparse"

    identical = select_test_candidates([record(1, 0.005, 10; id = "only")])
    @test identical.sparse === nothing
    no_sparse = select_test_candidates([record(1, 0.02, 10; id = "only")])
    @test no_sparse.sparse === nothing

    tied = select_test_candidates([
        record(2, 0.005, 20; id = "late", run = "a"),
        record(2, 0.005, 10; id = "z", run = "z"),
        record(2, 0.005, 10; id = "a", run = "a"),
    ])
    @test tied.match[:candidate_id] == "a"
end

@testset "Atomic worker states" begin
    mktempdir() do directory
        path = joinpath(directory, "status.jld2")
        @test load_status(path) === nothing
        write_status!(path; state = :running, update = 25)
        @test load_status(path)[:state] === :running
        write_status!(path; state = :failed, error_message = "synthetic")
        @test load_status(path)[:state] === :failed
        write_status!(path; state = :complete, update = 100)
        @test load_status(path)[:state] === :complete
        @test_throws Exception write_status!(path; state = :unknown)
    end
end

println("Package 6 study tests passed.")
