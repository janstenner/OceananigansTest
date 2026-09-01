using JLD2
using Test

include(joinpath(@__DIR__, "make_paper_figures.jl"))

function write_fixture_csv(path, configuration, candidate_id; qualified = true)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "run_id,replicate,configuration,strength,update,candidate_id,threshold_id,threshold_value,active_groups,active_inputs,validation_matching,pooled_pareto,under_quality_threshold")
        println(io, "fixture-r1,1,$configuration,0.01,0,native-$candidate_id,native,0.0,12,1152,$(qualified ? 0.005 : 0.04),false,$qualified")
        println(io, "fixture-r1,1,$configuration,0.01,25,$candidate_id,threshold_1,0.003,4,576,$(qualified ? 0.004 : 0.04),true,$qualified")
    end
end

function write_fixture_baseline(root, controller, value)
    path = joinpath(root, "varying", "$controller.jld2")
    mkpath(dirname(path))
    episode = (
        state_nusselt = fill(value / 200, 200),
        sum_state_nusselt = value,
    )
    JLD2.jldsave(
        path;
        status = :complete,
        protocol = :varying,
        controller,
        case_count = 8,
        expert_sha256 = controller === :expert ? "fixtureexpert" : "",
        episodes = fill(episode, 8),
    )
    return path
end

@testset "Package-8 paper artifacts" begin
    @test vec(panel_titles()) == [
        "GO - GC", "GO - SC", "GR - GC", "GR - SC",
        "Group Lasso - GC", "Group Lasso - SC", "GrOWL - GC", "GrOWL - SC",
    ]
    mktempdir() do directory
        latest_root = joinpath(directory, "latest")
        mkpath(joinpath(latest_root, "260101_010101", "go-gc"))
        mkpath(joinpath(latest_root, "260102_010101", "go-gc"))
        @test latest_experiment_id(latest_root) == "260102_010101"
        results = joinpath(directory, "results")
        experiment = "fixture"
        baseline_root = joinpath(directory, "baselines")
        output = joinpath(directory, "paper")
        previous = get(ENV, "REVISION_BASELINE_RESULTS_DIR", nothing)
        try
            ENV["REVISION_BASELINE_RESULTS_DIR"] = baseline_root
            write_fixture_baseline(baseline_root, :expert, 545.0)
            write_fixture_baseline(baseline_root, :unactuated, 800.0)
            for (index, configuration) in enumerate(P8_CONFIGURATION_NAMES)
                analysis = analysis_directory(results, experiment, configuration)
                mkpath(joinpath(analysis, "test"))
                candidate_id = "candidate-$index"
                has_candidate = configuration != "growl-gc"
                JLD2.jldsave(
                    joinpath(analysis, "status.jld2");
                    state = :complete,
                    experiment_id = experiment,
                    configuration,
                )
                write_fixture_csv(joinpath(analysis, "evaluations.csv"), configuration, candidate_id; qualified = has_candidate)
                write_fixture_csv(joinpath(analysis, "pooled_pareto_front.csv"), configuration, candidate_id; qualified = has_candidate)
                has_candidate || continue
                mask = falses(3, 48, 8)
                mask[1, :, :] .= true
                mask[2, 1:24, :] .= true
                candidate = Dict{Symbol, Any}(
                    :candidate_id => candidate_id,
                    :run_id => "fixture-r1",
                    :configuration => configuration,
                    :regularization_strength => 0.01,
                    :replicate => 1,
                    :update => 25,
                    :threshold_id => :threshold_1,
                    :threshold_value => 0.003,
                    :validation_matching => 0.004,
                    :active_groups => 4,
                    :active_inputs => count(mask),
                    :global_mask => BitArray(mask),
                )
                JLD2.jldsave(
                    joinpath(analysis, "selected_test_candidate.jld2");
                    frozen_before_test = true,
                    selection_uses_test_data = false,
                    quality_threshold = P8_QUALITY_THRESHOLD,
                    candidate,
                )
                episodes = [(
                    case_id = "fixture-$episode",
                    split = :test,
                    base_seed = episode <= 4 ? 101 : 202,
                    mirror = iseven(episode),
                    offset = episode % 2 == 0 ? 20 : 0,
                    evaluation_seed = P8_MASTER_SEED + episode,
                    episode,
                    state_nusselt = fill(2.75, 200),
                ) for episode in 1:8]
                JLD2.jldsave(
                    joinpath(analysis, "test", "test_results.jld2");
                    candidate_id,
                    active_inputs = count(mask),
                    validation_matching = 0.004,
                    expert_identifier = "sha256:fixtureexpert",
                    case_count = 8,
                    episodes,
                )
            end
            checked = main([
                "--experiment-id", experiment,
                "--results-dir", results,
                "--output-dir", output,
                "--check-only",
            ])
            @test count(data -> !isnothing(data.selected), values(checked.configurations)) == 7
            automatic = main([
                "--results-dir", results,
                "--output-dir", output,
                "--check-only",
            ])
            @test automatic.options.experiment_id == experiment
            artifacts = main([
                "--experiment-id", experiment,
                "--results-dir", results,
                "--output-dir", output,
            ])
            @test length(artifacts.rows) == 10
            @test artifacts.rows[1].configuration == "Full sensor set expert"
            @test artifacts.rows[1].mean_state_nusselt ≈ 2.725
            @test artifacts.rows[end].configuration == "Unactuated"
            @test artifacts.rows[end].mean_state_nusselt == 4.0
            go_gc = only(row for row in artifacts.rows if row.configuration == "go-gc")
            @test go_gc.mean_state_nusselt == 2.75
            @test go_gc.global_sc_sparsity_percent == 50.0
            @test go_gc.global_gc_sparsity_percent == 0.0
            growl_gc = only(row for row in artifacts.rows if row.configuration == "growl-gc")
            @test growl_gc.active_groups === missing
            @test growl_gc.minimum_native_active_groups_under_quality_threshold === missing
            @test all(isfile, artifacts.mask_paths)
            @test length(artifacts.mask_paths) == 6
            @test all(isfile, artifacts.pareto_paths)
            @test all(filesize(path) > 0 for path in vcat(artifacts.mask_paths, artifacts.pareto_paths))
            @test length(readlines(artifacts.table.csv_path)) == 11
            @test occursin("go-gc,4/32,", read(artifacts.table.csv_path, String))
            @test occursin("| go-sc | 4/96 |", read(artifacts.table.markdown_path, String))
            @test occursin("Full sensor set expert", read(artifacts.table.markdown_path, String))
            @test isfile(joinpath(output, "paper_metrics.jld2"))
        finally
            if isnothing(previous)
                delete!(ENV, "REVISION_BASELINE_RESULTS_DIR")
            else
                ENV["REVISION_BASELINE_RESULTS_DIR"] = previous
            end
        end
    end
end

println("package8-paper-figure-tests-ok")
