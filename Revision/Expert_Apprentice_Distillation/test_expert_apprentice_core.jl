ENV["DISTILLATION_PROTOCOL"] = "fixed"
ENV["DISTILLATION_ALLOW_FRESH_EXPERT"] = "true"
ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"

using Random
using Test

include(joinpath(@__DIR__, "Expert_Apprentice.jl"))

@testset "canonical apprentice methods" begin
    @test available_apprentice_kinds() == [:go, :gr, :group_lasso, :growl]
    @test_throws ErrorException normalize_apprentice_kind(:gro_asc)
    @test_throws ErrorException normalize_apprentice_kind(:weighted)
end

@testset "pure group-consistent candidates" begin
    test_model = deepcopy(apprentice)
    groups = regularizer_groups(test_model; group_rows_by_overlap = true, group_channels = true)
    @test !any_shared(groups)
    test_model.encoder.embedding.weight[:, groups[1]] .= 0

    specifications = [
        HardThresholdSpec(:relative_half, :relative_to_max, 0.5),
        HardThresholdSpec(:keep_four, :keep_groups, 4),
    ]
    candidates = candidate_masks(test_model, specifications; groups)
    @test getindex.(candidates, :threshold_id) == [:native, :relative_half, :keep_four]
    @test candidates[1][:threshold_kind] == :native
    @test candidates[2][:threshold_kind] == :hard_threshold
    @test candidates[3][:active_groups] == 4
    @test candidates[1][:group_mask][1] == false
    @test all(candidate -> candidate[:active_inputs] <= 3 * 48 * 8, candidates)
    @test all(candidate -> size(candidate[:global_mask]) == (3, 48, 8), candidates)
end

@testset "regularizer names and proximal interval primitives" begin
    weights = Float32[3 0 1 0; 0 4 0 2]
    groups = [[1, 2], [3, 4]]
    original_norm = norm(weights)
    apply_grouped_regularizer!(
        weights;
        groups,
        regularization_strength = 0.1,
        theta_mode = :group_lasso,
    )
    @test norm(weights) < original_norm

    operator_weights = ones(Float32, 2)
    apply_group_reweighted_regularizer!(
        weights;
        groups,
        operator_weights,
        regularization_strength = 0.1,
    )
    @test all(isfinite, weights)
end

@testset "corpus validation, training, and archive integration" begin
    sample_count = 4
    observations = rand(MersenneTwister(11), Float32, 3, 48, 8, sample_count)
    local_observations = local_mat_observation_batch(
        observations;
        actuator_sensor_indices = actuators_to_sensors,
        window_size,
    )
    expert_actions = prob(agent.policy, local_observations, nothing).μ
    dataset = Dict{Symbol, Any}(
        :observations => observations,
        :expert_actions => Float32.(expert_actions),
        :sample_count => sample_count,
        :source_files => ["synthetic-test"],
    )
    dense_mask = ones(Float32, size(local_observations, 1))
    @test isfinite(evaluate_expert_matching(
        apprentice,
        dataset,
        dense_mask;
        batch_size = 2,
        prediction_mode = :autoregressive,
    ))

    run_directory = mktempdir()
    try
        schedule = CandidateSchedule(
            start_update = 0,
            evaluation_interval = 1,
            garbage_collection_interval = 2,
            resume_interval = 1,
        )
        manager = initialize_pareto_archive(
            run_directory;
            run_id = "expert-apprentice-core-test",
            schedule,
            config = Dict(:purpose => :test),
        )
        training_config = ApprenticeTrainingConfig(
            regularized_updates = 2,
            post_pruning_finetune_updates = 1,
            batch_size = 2,
            proximal_interval = 1,
            reweight_interval = 1,
            regularization_strength = 1e-5,
            validation_batch_size = 2,
            validation_prediction_mode = :autoregressive,
            diagnostic_teacher_forced = true,
        )
        result = train_apprentice!(
            deepcopy(apprentice);
            method = :go,
            train_dataset = dataset,
            validation_dataset = dataset,
            config = training_config,
            archive_manager = manager,
            threshold_specs = [HardThresholdSpec(:later_half, :relative_to_max, 0.5)],
            training_rng = MersenneTwister(17),
        )
        @test length(result.losses) == 3
        @test model_is_finite(result.model)
        @test manager.last_evaluated_update == 3
        @test isfile(latest_resume_path(manager))
        @test load_resume_checkpoint(manager).status == :complete
        @test !isempty(manager.front)
        @test all(candidate -> candidate[:loadable], manager.front)
    finally
        rm(run_directory; recursive = true, force = true)
    end
end

println("expert-apprentice-core-tests-ok")
