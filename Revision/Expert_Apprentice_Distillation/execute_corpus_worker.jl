# This file is deliberately included only after the selected Revision MAT run
# file. Julia 1.12 world-age rules then make all run-file globals and methods
# directly available to the worker adapter.
function execute_loaded_corpus_worker(options)
    protocol = options["protocol"]
    split = Symbol(lowercase(string(options["split"])))
    base_seed = options["base_seed"]
    mirror = something(options["mirror"], false)
    worker_directory = abspath(string(options["worker_dir"]))

    expert_metadata = load_distillation_expert!(
        protocol;
        explicit_path = options["expert_path"],
        allow_fresh_expert = options["allow_fresh_expert"],
    )

    initialize_episode! = if protocol === :fixed
        _ -> generate_random_init()
    else
        spec -> generate_random_init(
            split = spec.split,
            base_seed = spec.base_seed,
            mirror = spec.mirror,
            offset = spec.offset,
        )
    end
    observe = () -> global_sensor_observation(
        env.y;
        horizontal_indices = sensor_positions[1],
        vertical_indices = sensor_positions[2],
        add_joon_position_encoding = joon_pe,
    )
    expert_mean = () -> RL.prob(agent.policy, env.state, nothing).μ
    advance! = action -> env(action)
    observation_metadata = Dict{Symbol, Any}(
        :channels => DISTILLATION_CHANNELS,
        :horizontal_sensors => DISTILLATION_HORIZONTAL_SENSORS,
        :vertical_sensors => DISTILLATION_VERTICAL_SENSORS,
        :sensor_positions => deepcopy(sensor_positions),
        :actuator_sensor_indices => copy(actuators_to_sensors),
        :window_size => window_size,
        :joon_position_encoding => joon_pe,
        :temporal_steps => temporal_steps,
        :memory_size => memory_size,
        :control_dt => dt,
    )

    offsets = if isnothing(options["offset_start"])
        distillation_offsets(protocol, split)
    else
        collect(options["offset_start"]:(options["offset_start"] + options["offset_count"] - 1))
    end
    verification_spec = protocol === :fixed ?
        (split = :shared, base_seed = nothing, mirror = false, offset = nothing) :
        (split = split, base_seed = base_seed, mirror = mirror, offset = first(offsets))
    initialize_episode!(verification_spec)
    assert_lossless_observation_reconstruction(
        observe(),
        env.state;
        actuator_sensor_indices = actuators_to_sensors,
        window_size,
    )

    output = generate_distillation_worker!(
        ;
        protocol,
        split,
        base_seed,
        mirror,
        offsets,
        rollout_steps = options["steps"],
        initialize_episode!,
        observe,
        expert_mean,
        advance!,
        expert_metadata,
        observation_metadata,
        run_seed = options["run_seed"],
        worker_directory,
        overwrite = options["overwrite"],
    )
    println("Worker completed: $output")
    return output
end
