using StableRNGs
using Test

include(joinpath(@__DIR__, "NoiseStudy.jl"))
using .NoiseStudy

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

@testset "Noise Study job and seed plan" begin
    @test length(job_records(:fixed)) == 30
    @test length(job_records(:varying)) == 30
    @test count(job -> job.noise_level == 0, job_records(:fixed)) == 3
    @test all(job -> job.replicate_count == 10, filter(job -> job.noise_level > 0, job_records(:fixed)))
    @test level_tag.(NOISE_LEVELS) == (
        "n_0p00", "n_0p01", "n_0p05", "n_0p10",
        "n_0p20", "n_0p30", "n_0p40", "n_0p50", "n_0p70", "n_1p00",
    )
    paired = noise_seed(:varying, 0.05, 3, 7)
    @test paired == noise_seed(:varying, 0.05, 3, 7)
    @test paired != noise_seed(:varying, 0.05, 4, 7)
    @test paired != noise_seed(:varying, 0.10, 3, 7)
    @test_throws ArgumentError noise_seed(:fixed, 0.0, 1, 1)
end

@testset "Physical white noise precedes positional encoding" begin
    physical = zeros(Float32, 3, 48, 8)
    scales = [2.0, 3.0, 4.0]
    first_result = noisy_encoded_global_observation(physical, StableRNG(123), 0.10, scales)
    second_result = noisy_encoded_global_observation(physical, StableRNG(123), 0.10, scales)
    other_result = noisy_encoded_global_observation(physical, StableRNG(124), 0.10, scales)
    @test first_result == second_result
    @test first_result != other_result
    clean_encoded = noisy_encoded_global_observation(physical, StableRNG(123), 0.0, scales)
    @test all(iszero, clean_encoded[2:3, :, :])
    for horizontal_index in 1:48
        @test all(clean_encoded[1, horizontal_index, :] .== Float32(sin(2pi * horizontal_index / 48)))
    end
end

@testset "Frozen validation-only controller selection" begin
    defaults = source_defaults(PROJECT_ROOT)
    fixed = select_sparse_sc_candidate(:fixed, defaults.package7_results)
    varying = select_sparse_sc_candidate(:varying, defaults.package8_results)
    @test fixed.configuration == "go-sc"
    @test varying.configuration == "gr-sc"
    @test fixed.candidate[:active_inputs] == minimum(row.active_inputs for row in fixed.selection_audit)
    @test varying.candidate[:active_inputs] == minimum(row.active_inputs for row in varying.selection_audit)
    @test count(row -> row.selected, fixed.selection_audit) == 1
    @test count(row -> row.selected, varying.selection_audit) == 1

    fixed_match = load_c_match_candidate(:fixed, defaults.go_study_results)
    varying_match = load_c_match_candidate(:varying, defaults.go_study_results)
    @test Symbol(fixed_match.candidate[:selection_role]) === :C_match
    @test Symbol(varying_match.candidate[:selection_role]) === :C_match
    @test length(fixed_match.candidate[:mask]) == 360
    @test length(varying_match.candidate[:mask]) == 360
end
