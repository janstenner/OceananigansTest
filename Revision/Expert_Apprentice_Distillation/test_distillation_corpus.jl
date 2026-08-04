ENV["DISTILLATION_SKIP_AUTOLOAD"] = "true"
include(joinpath(@__DIR__, "DistillationCorpus.jl"))

function run_distillation_corpus_tests()
    global current_dummy_offset = 0
    global current_dummy_step = 0
    horizontal_indices = collect(1:DISTILLATION_HORIZONTAL_SENSORS)
    vertical_indices = collect(1:DISTILLATION_VERTICAL_SENSORS)
    actuator_indices = collect(3:4:DISTILLATION_HORIZONTAL_SENSORS)

    raw_fields = reshape(
        Float32.(1:(DISTILLATION_CHANNELS * DISTILLATION_HORIZONTAL_SENSORS * DISTILLATION_VERTICAL_SENSORS)),
        DISTILLATION_CHANNELS,
        DISTILLATION_HORIZONTAL_SENSORS,
        DISTILLATION_VERTICAL_SENSORS,
    )
    global_observation = global_sensor_observation(
        raw_fields;
        horizontal_indices,
        vertical_indices,
        add_joon_position_encoding = false,
    )
    local_observation = local_mat_observation(
        global_observation;
        actuator_sensor_indices = actuator_indices,
    )
    @assert size(local_observation) == (360, 12)
    @assert assert_lossless_observation_reconstruction(
        global_observation,
        local_observation;
        actuator_sensor_indices = actuator_indices,
    )

    mktempdir() do worker_directory
        initialize_episode! = spec -> begin
            global current_dummy_offset = isnothing(spec.offset) ? 0 : spec.offset
            global current_dummy_step = 0
        end
        observe = () -> fill(
            Float32(current_dummy_offset * 10 + current_dummy_step),
            DISTILLATION_CHANNELS,
            DISTILLATION_HORIZONTAL_SENSORS,
            DISTILLATION_VERTICAL_SENSORS,
        )
        expert_mean = () -> fill(Float32(current_dummy_offset), 1, DISTILLATION_ACTUATORS)
        advance! = _ -> (global current_dummy_step += 1)
        metadata = Dict{Symbol, Any}(:identifier => "dummy-expert")

        varying_path = generate_distillation_worker!(
            ;
            protocol = :varying,
            split = :train,
            base_seed = 123,
            mirror = true,
            offsets = 0:1,
            rollout_steps = 3,
            initialize_episode!,
            observe,
            expert_mean,
            advance!,
            expert_metadata = metadata,
            worker_directory,
        )
        @assert isfile(varying_path)

        fixed_path = generate_distillation_worker!(
            ;
            protocol = :fixed,
            rollout_steps = 2,
            initialize_episode!,
            observe,
            expert_mean,
            advance!,
            expert_metadata = metadata,
            worker_directory,
        )
        @assert isfile(fixed_path)

        corpus = load_distillation_corpus(worker_directory)
        varying = corpus[:varying][:train]
        @assert varying[:sample_count] == 6
        @assert varying[:worker_count] == 1
        @assert length(varying[:episodes]) == 2
        @assert size(varying[:observations]) == (3, 48, 8, 6)
        @assert varying[:episodes][2][:sample_start] == 4
        batch = distillation_batch(
            varying,
            [2, 5];
            actuator_sensor_indices = actuator_indices,
        )
        @assert size(batch.observations) == (360, 12, 2)
        @assert size(batch.expert_actions) == (1, 12, 2)
        @assert batch.sample_indices == [2, 5]
        @assert corpus[:fixed][:train] === corpus[:fixed][:validation]
        @assert corpus[:fixed][:validation] === corpus[:fixed][:test]
        @assert corpus[:fixed][:train][:sample_count] == 2
        @assert !corpus[:fixed][:train][:coverage_complete]
        @assert expected_distillation_counts(:varying, :train) == (
            workers = 40,
            episodes = 3840,
            samples = 768000,
        )
    end
    println("distillation-corpus-tests-ok")
end

run_distillation_corpus_tests()
