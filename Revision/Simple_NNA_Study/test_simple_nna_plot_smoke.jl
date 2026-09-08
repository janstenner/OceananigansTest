using Test

include(joinpath(@__DIR__, "analyze_configuration_worker.jl"))

@testset "Simple-NNA pooled Pareto plot" begin
    global CORPUS = Dict(:test => Dict(101 => nothing, 202 => nothing))
    cases = varying_test_cases()
    @test length(cases) == 8
    @test Set(case.base_seed for case in cases) == Set((101, 202))
    @test Set(case.mirror for case in cases) == Set((false, true))
    @test Set(case.offset for case in cases) == Set((0, 20))
    @test length(unique(case.evaluation_seed for case in cases)) == 8
    records = Dict{Symbol, Any}[]
    candidate_index = 0
    fixture_thresholds = (0.0, 0.004, 0.009, 0.02)
    for replicate in SNN_REPLICATES, (threshold_index, threshold) in enumerate(fixture_thresholds)
        replicate == 1 && threshold == 0.02 && continue
        candidate_index += 1
        push!(records, Dict{Symbol, Any}(
            :run_id => "smoke-r$replicate",
            :candidate_id => "candidate-$candidate_index",
            :replicate => replicate,
            :configuration => "go-sc",
            :regularization_strength => 0.09,
            :update => 25 * threshold_index,
            :threshold_id => threshold == 0 ? :native : Symbol("threshold_$threshold_index"),
            :threshold_value => threshold,
            :active_groups => 20 - 2 * threshold_index,
            :active_inputs => 500 - 30 * threshold_index - replicate,
            :validation_matching => 1e-3 * (1 + threshold_index / 2 + replicate / 10),
            :numeric_status => :ok,
            :pareto_scope => :simple_nna_thresholds,
        ))
    end
    front = pareto_front(records)
    @test !isempty(front)
    @test length(records) == 11
    @test observed_thresholds(records) == collect(fixture_thresholds)
    @test Set(keys(threshold_colors(observed_thresholds(records)))) == Set(fixture_thresholds)
    filter_fixture = Dict{Symbol, Any}[
        Dict(:update => 0, :threshold_id => :native, :active_groups => 5),
        Dict(:update => 0, :threshold_id => :same_groups, :active_groups => 5),
        Dict(:update => 0, :threshold_id => :fewer_groups, :active_groups => 4),
    ]
    filtered = retain_successful_threshold_records(filter_fixture; context = "smoke")
    @test Symbol.(getindex.(filtered, :threshold_id)) == [:native, :fewer_groups]
    @test observed_strengths(records) == [0.09]
    sparse = select_sparse_test_candidate(front)
    @test sparse[:validation_matching] <= SNN_QUALITY_THRESHOLD
    @test sparse[:active_inputs] == minimum(
        record[:active_inputs] for record in front
        if record[:validation_matching] <= SNN_QUALITY_THRESHOLD
    )
    @test isnothing(select_sparse_test_candidate([
        Dict{Symbol, Any}(
            :validation_matching => 2 * SNN_QUALITY_THRESHOLD,
            :active_inputs => 1,
            :active_groups => 1,
            :update => 0,
            :run_id => "nr",
            :candidate_id => "nr",
        ),
    ]))
    gr_sc_front = Dict{Symbol, Any}[
        Dict(:validation_matching => 0.015, :active_inputs => 30, :active_groups => 3,
             :update => 25, :run_id => "gr-sc", :candidate_id => "g3"),
        Dict(:validation_matching => 0.010, :active_inputs => 40, :active_groups => 4,
             :update => 50, :run_id => "gr-sc", :candidate_id => "g4"),
        Dict(:validation_matching => 0.008, :active_inputs => 50, :active_groups => 5,
             :update => 75, :run_id => "gr-sc", :candidate_id => "g5"),
        Dict(:validation_matching => 0.004, :active_inputs => 170, :active_groups => 17,
             :update => 100, :run_id => "gr-sc", :candidate_id => "g17"),
        Dict(:validation_matching => 0.003, :active_inputs => 180, :active_groups => 18,
             :update => 125, :run_id => "gr-sc", :candidate_id => "g18"),
    ]
    gr_sc_selections = select_quality_candidates(gr_sc_front, "gr-sc")
    @test length(gr_sc_selections.threshold_selections) == 1
    @test length(gr_sc_selections.unique_candidates) == 4
    @test only(gr_sc_selections.threshold_selections)[:candidate_id] == "g3"
    @test [entry[:candidate][:candidate_id] for entry in gr_sc_selections.unique_candidates] ==
          ["g3", "g4", "g5", "g17"]
    @test gr_sc_selections.pareto_sweep_candidate_indices == [1, 2, 3, 4]
    @test gr_sc_selections.unique_candidates[1][:quality_thresholds] == [0.02]
    @test all(isempty(entry[:quality_thresholds]) for entry in gr_sc_selections.unique_candidates[2:end])
    mktempdir() do directory
        options = (configuration = "go-sc", strengths = [999.0])
        selections = select_quality_candidates(front, options.configuration)
        paths = make_plot(options, records, front, selections, directory)
        @test length(paths) == 2
        @test all(isfile, paths)
        @test all(filesize(path) > 0 for path in paths)
        svg = read(first(paths), String)
        @test occursin("λ ∈ {0.09}", svg)
        @test !occursin("λ ∈ {999", svg)
        @test occursin(">τ=0.02</text>", svg)
        @test occursin("stroke-dasharray", svg)
        front_ids = Set(string(record[:candidate_id]) for record in front)
        front_csv = write_csv(joinpath(directory, "pooled_pareto_front.csv"), front, front_ids)
        front_lines = readlines(front_csv)
        @test length(front_lines) == length(front) + 1
        @test endswith(first(front_lines), ",qualified_quality_thresholds")
        mkpath(joinpath(directory, "test"))
        episode = (
            case_id = "fixture",
            split = :test,
            base_seed = 101,
            mirror = false,
            offset = 0,
            evaluation_seed = SNN_MASTER_SEED + 1,
            episode = 1,
            rewards = collect(range(-3.0, -2.0; length = SNN_TEST_STEPS)),
            state_nusselt = collect(range(3.0, 2.0; length = SNN_TEST_STEPS)),
            simulation_times = collect(1.0:SNN_TEST_STEPS),
            actions = zeros(Float32, SNN_TEST_STEPS, 12),
        )
        test_csv = write_test_csv(joinpath(directory, "test", "test_episodes.csv"), [episode])
        @test length(readlines(test_csv)) == SNN_TEST_STEPS + 1
        @test startswith(readline(test_csv), "case,split,base_seed,mirror,offset,evaluation_seed,episode,step,simulation_time")
        test_plot = make_test_plot(directory, [episode], sparse, [0.02])
        @test isfile(test_plot)
        @test filesize(test_plot) > 0
        stale = joinpath(directory, "selected_test_candidates.jld2")
        write(stale, "stale")
        clear_selected_candidate_test!(directory)
        @test !isfile(stale)
    end
end

println("simple_nna-plot-smoke-ok")
