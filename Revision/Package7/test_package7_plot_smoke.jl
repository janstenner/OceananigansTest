using Test

include(joinpath(@__DIR__, "analyze_configuration_worker.jl"))

@testset "Package-7 pooled Pareto plot" begin
    records = Dict{Symbol, Any}[]
    candidate_index = 0
    fixture_thresholds = (0.0, 0.004, 0.009, 0.02)
    for replicate in P7_REPLICATES, (threshold_index, threshold) in enumerate(fixture_thresholds)
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
            :pareto_scope => :package7_thresholds,
        ))
    end
    front = pareto_front(records)
    @test !isempty(front)
    @test length(records) == 12
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
    mktempdir() do directory
        options = (configuration = "go-sc", strengths = [999.0])
        paths = make_plot(options, records, front, directory)
        @test length(paths) == 2
        @test all(isfile, paths)
        @test all(filesize(path) > 0 for path in paths)
        svg = read(first(paths), String)
        @test occursin("λ ∈ {0.09}", svg)
        @test !occursin("λ ∈ {999", svg)
        front_ids = Set(string(record[:candidate_id]) for record in front)
        front_csv = write_csv(joinpath(directory, "pooled_pareto_front.csv"), front, front_ids)
        @test length(readlines(front_csv)) == length(front) + 1
    end
end

println("package7-plot-smoke-ok")
