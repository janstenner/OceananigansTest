# Included only after the selected Higher-Ra MAT run file. This keeps access to
# dynamically defined run-file globals world-age safe on Julia 1.12.
function execute_loaded_higher_ra_corpus_worker(options)
    study = options["study"]
    split = options["split"]
    base_seed = options["base_seed"]
    mirror = options["mirror"]
    worker_directory = options["worker_dir"]
    source = validate_higher_ra_sources(study; expert_path = options["expert_path"])

    Float64(RA) == study.rayleigh || error(
        "Loaded run file has Ra=$(Float64(RA)); expected $(study.rayleigh).",
    )
    abspath(CORPUS_PATH) == source.state_corpus_path || error(
        "Loaded run file uses the wrong state corpus: $(abspath(CORPUS_PATH)).",
    )
    length(CORPUS[split]) == DISTILLATION_EXPECTED_BASES[split] || error(
        "Loaded $(study.label) $split split has unexpected basis-snapshot coverage.",
    )

    expert_metadata = load_distillation_expert!(
        :varying;
        explicit_path = source.expert_path,
        allow_fresh_expert = false,
    )
    expert_metadata[:higher_ra_study] = study.tag
    expert_metadata[:rayleigh] = study.rayleigh
    expert_metadata[:state_corpus_path] = source.state_corpus_path
    expert_metadata[:state_corpus_sha256] = source.state_corpus_sha256
    expert_metadata[:run_file_path] = source.run_file_path
    expert_metadata[:run_file_sha256] = source.run_file_sha256

    initialize_episode! = spec -> generate_random_init(
        split = spec.split,
        base_seed = spec.base_seed,
        mirror = spec.mirror,
        offset = spec.offset,
    )
    observe = () -> global_sensor_observation(
        env.y;
        horizontal_indices = sensor_positions[1],
        vertical_indices = sensor_positions[2],
        add_joon_position_encoding = joon_pe,
    )
    expert_mean = () -> RL.prob(agent.policy, env.state, nothing).μ
    advance! = action -> env(action)
    observation_metadata = Dict{Symbol, Any}(
        :higher_ra_study => study.tag,
        :rayleigh => study.rayleigh,
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

    offsets = distillation_offsets(:varying, split)
    verification_spec = (;
        split,
        base_seed,
        mirror,
        offset = first(offsets),
    )
    initialize_episode!(verification_spec)
    assert_lossless_observation_reconstruction(
        observe(),
        env.state;
        actuator_sensor_indices = actuators_to_sensors,
        window_size,
    )

    output = generate_distillation_worker!(
        ;
        protocol = :varying,
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
