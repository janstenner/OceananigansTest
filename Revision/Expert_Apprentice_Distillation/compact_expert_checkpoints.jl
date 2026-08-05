using JLD2
using Printf

# Include this file only from an already initialized MAT REPL. Loading the
# agent requires the MAT/RL types to be defined in the current Julia session.
# Merely including the file compacts both selected expert checkpoints in place.
const EXPERT_CHECKPOINTS_TO_COMPACT = [
    joinpath(@__DIR__, "experts", "fixed", "agent.jld2"),
    joinpath(@__DIR__, "experts", "varying", "agent.jld2"),
]

function human_bytes(bytes::Integer)
    value = Float64(bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 1
    while value >= 1024 && unit_index < length(units)
        value /= 1024
        unit_index += 1
    end
    return @sprintf("%.2f %s", value, units[unit_index])
end

function checkpoint_keys(path::AbstractString)
    return JLD2.jldopen(path, "r") do file
        sort!(String.(collect(keys(file))))
    end
end

function empty_and_shrink_trajectory!(agent)
    hasproperty(agent, :trajectory) || error(
        "The saved agent has no trajectory field.",
    )
    trajectory = agent.trajectory
    empty!(trajectory)

    for (trace_name, trace) in Base.pairs(trajectory.traces)
        hasproperty(trace, :buffer) || error(
            "Trajectory trace '$trace_name' has no shrinkable backing buffer.",
        )
        old_buffer = trace.buffer
        dimensions = ntuple(
            dimension -> dimension == ndims(old_buffer) ? 1 : size(old_buffer, dimension),
            ndims(old_buffer),
        )
        trace.buffer = similar(old_buffer, dimensions)
        trace.first = 1
        trace.nframes = 0
    end

    isempty(trajectory) || error("Trajectory is not empty after compaction.")
    all(
        size(trace.buffer, ndims(trace.buffer)) == 1
        for trace in Base.values(trajectory.traces)
    ) ||
        error("At least one trajectory backing buffer was not reduced to capacity 1.")
    return agent
end

function compact_selected_expert_checkpoints!()
    # Deserialize both agents before touching either source file. If a required
    # type is missing in the REPL, the operation therefore fails without
    # modifying one checkpoint and leaving the other untouched.
    selected = map(EXPERT_CHECKPOINTS_TO_COMPACT) do path
        checkpoint_path = abspath(path)
        isfile(checkpoint_path) || error("Checkpoint does not exist: $checkpoint_path")
        original_keys = checkpoint_keys(checkpoint_path)
        "agent" in original_keys || error(
            "Checkpoint has no top-level 'agent' entry: $checkpoint_path",
        )
        agent = JLD2.load(checkpoint_path, "agent")
        empty_and_shrink_trajectory!(agent)
        (
            path = checkpoint_path,
            agent,
            original_keys,
            original_bytes = filesize(checkpoint_path),
        )
    end

    temporary_paths = String[]
    try
        # Likewise, prepare and fully validate both replacements before either
        # original checkpoint is replaced.
        for checkpoint in selected
            temporary_path = joinpath(
                dirname(checkpoint.path),
                ".$(basename(checkpoint.path)).agent-only.tmp.$(getpid()).$(time_ns())",
            )
            push!(temporary_paths, temporary_path)
            JLD2.jldsave(temporary_path; agent = checkpoint.agent)
            checkpoint_keys(temporary_path) == ["agent"] || error(
                "Compacted checkpoint contains unexpected entries: $temporary_path",
            )
            compacted_agent = JLD2.load(temporary_path, "agent")
            isempty(compacted_agent.trajectory) || error(
                "Reloaded compacted agent contains a non-empty trajectory: $temporary_path",
            )
            all(
                size(trace.buffer, ndims(trace.buffer)) == 1
                for trace in Base.values(compacted_agent.trajectory.traces)
            ) || error(
                "Reloaded compacted agent has a trajectory buffer with capacity above 1: " *
                temporary_path,
            )
        end

        for (checkpoint, temporary_path) in zip(selected, temporary_paths)
            mv(temporary_path, checkpoint.path; force = true)
            compacted_bytes = filesize(checkpoint.path)
            removed_count = length(setdiff(checkpoint.original_keys, ["agent"]))
            println("Compacted: $(checkpoint.path)")
            println("  removed top-level entries: $removed_count")
            println(
                "  size: $(human_bytes(checkpoint.original_bytes)) -> " *
                human_bytes(compacted_bytes),
            )
        end
    finally
        for temporary_path in temporary_paths
            isfile(temporary_path) && rm(temporary_path; force = true)
        end
    end
    return nothing
end

compact_selected_expert_checkpoints!()
