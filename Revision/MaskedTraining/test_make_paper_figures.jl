using JLD2
using Test

include("make_paper_figures.jl")

function write_result(path, entry, protocol, rewards; identity = nothing)
    mkpath(dirname(path))
    pairs = Dict{Symbol, Any}(
        :status => "complete",
        :episode_target => length(rewards),
        :episodes_completed => length(rewards),
        :rewards => rewards,
        :run_seed => entry.run_seed,
        :ic_seed => entry.ic_seed,
    )
    isnothing(identity) || (pairs[:manifest_identity] = identity)
    JLD2.jldsave(path; pairs...)
end

@testset "MaskedTraining paper curves" begin
    @test SERIES_COLORS == Dict(:dense => "#277DA1", :gc => "#F2A13A", :sc => "#B41A5C")
    @test rolling_mean(collect(1.0:51.0)) == [25.5, 26.5]
    mktempdir() do root
        results = joinpath(root, "masked")
        comparison = joinpath(root, "comparison")
        entries = [(run_id = "run-$index", run_seed = 100 + index, ic_seed = 200 + index) for index in 1:10]
        manifest = (; identity = "fixture-identity", entries)
        mkpath(results)
        JLD2.jldsave(joinpath(results, "manifest.jld2"); manifest)
        for protocol in PROTOCOLS, entry in entries
            count = EPISODE_TARGETS[protocol]
            dense = fill(-600.0, count)
            write_result(joinpath(comparison, "runs", entry.run_id, string(protocol), "mat.jld2"), entry, protocol, dense)
            write_result(joinpath(results, "runs", string(protocol), "gc", "$(entry.run_id).jld2"), entry, protocol, dense .+ 10; identity = manifest.identity)
            write_result(joinpath(results, "runs", string(protocol), "sc", "$(entry.run_id).jld2"), entry, protocol, dense .+ 20; identity = manifest.identity)
        end
        curves = load_curves(results, comparison)
        @test all(length(curves[(protocol, series)]) == 10 for protocol in PROTOCOLS for series in SERIES)
        @test aggregate_curves(curves[(:fixed, :dense)]).episodes == collect(50:2000)
        plot = learning_curves_combined(curves)
        @test length(plot.plot.data) == 87 # 2 panels x 3 x (IQR pair + 10 runs + median + mean) + 3 style keys
        @test count(trace -> get(trace.fields, :showlegend, true), plot.plot.data) == 6
        rows = final_100_statistics(curves)
        @test length(rows) == 6
        @test only(row for row in rows if row.protocol === :fixed && row.series === :gc).mean_paired_difference_from_dense == 10
        @test only(row for row in rows if row.protocol === :varying && row.series === :sc).paired_wins == 10
        csv = write_statistics_csv(joinpath(root, "statistics.csv"), rows)
        @test length(readlines(csv)) == 7
        broken = joinpath(results, "runs", "fixed", "gc", "run-1.jld2")
        write_result(broken, first(entries), :fixed, fill(-590.0, EPISODE_TARGETS[:fixed]); identity = "wrong")
        @test_throws ErrorException load_curves(results, comparison)
    end
end
