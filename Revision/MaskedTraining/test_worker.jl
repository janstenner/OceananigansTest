include("run_worker.jl")
using Test

protocol = Symbol(only(ARGS))
m = S.build_manifest()
entry = first(filter(e -> e.origin !== :imported_package3, m.entries))
mktempdir() do results
    S.atomic_save(S.manifest_path(results); manifest = m)
    path = run_worker(; results, protocol, grouping = :gc, run_id = entry.run_id, episodes = 0)
    @test S.complete_result(path, m.identity, 0)
    @test !isdir(path * ".lock")
    JLD2.jldopen(path, "r") do f
        @test f["run_seed"] == entry.run_seed
        @test f["ic_seed"] == entry.ic_seed
        @test f["control_steps"] == 0
        @test f["reward_uses_full_sensors"]
        @test f["run_parameters"]["Ra"] == 1e4
        @test f["initial_parameter_hash"] == m.baselines[(entry.run_id, protocol)].initial_hash
    end
    digest = S.file_sha256(path)
    @test run_worker(; results, protocol, grouping = :gc, run_id = entry.run_id, episodes = 0) == path
    @test S.file_sha256(path) == digest
    @test_throws ErrorException S.complete_result(path, m.identity, S.BUDGETS[protocol])
    println("$protocol: generated-seed pairing, atomic worker save and completed-run reuse passed.")
end
