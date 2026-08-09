using JLD2
using PlotlyJS
using Printf

const P6_CALIBRATION_INSPECTION_ROOT = joinpath(
    @__DIR__,
    "results",
    "strength_calibration",
)
const P6_CALIBRATION_ANALYSIS_DIRECTORY = joinpath(
    P6_CALIBRATION_INSPECTION_ROOT,
    "analysis",
)
const P6_CALIBRATION_PLOT_COMBINATIONS = (
    (protocol = :fixed, grouping = :grouped_channels),
    (protocol = :fixed, grouping = :separate_channels),
    (protocol = :varying, grouping = :grouped_channels),
    (protocol = :varying, grouping = :separate_channels),
)
const P6_CALIBRATION_STRENGTH_COLORS = (
    "#2166AC",
    "#4393C3",
    "#92C5DE",
    "#7B3294",
    "#F4A582",
    "#D6604D",
    "#B2182B",
    "#67001F",
)
const P6_CALIBRATION_UPDATES = Dict(:fixed => 35_000, :varying => 50_000)
const P6_CALIBRATION_REGRESSION_LEARNING_RATE = 2e-4
const P6_CALIBRATION_PHASE = :long_budget_rerun
const P6_CALIBRATION_STRENGTHS = Dict(
    (:fixed, :grouped_channels) => [0.003, 0.006, 0.01, 0.03, 0.06, 0.09],
    (:fixed, :separate_channels) => [0.0015, 0.003, 0.006, 0.01, 0.02, 0.03],
    (:varying, :grouped_channels) => [0.003, 0.008, 0.025, 0.04, 0.06],
    (:varying, :separate_channels) => [0.003, 0.008, 0.025, 0.04, 0.06],
)
const P6_CALIBRATION_TEST_WORKER = joinpath(
    @__DIR__,
    "inspect_strength_calibration_test_worker.jl",
)

function calibration_value(container, key::Symbol)
    haskey(container, key) && return container[key]
    haskey(container, string(key)) && return container[string(key)]
    error("Calibration result has no '$key' entry.")
end

function calibration_run_directories(protocol::Symbol, grouping::Symbol)
    directory = joinpath(
        P6_CALIBRATION_INSPECTION_ROOT,
        string(protocol),
        string(grouping),
    )
    isdir(directory) || error("Calibration combination is missing: $directory")
    runs = filter(
        path -> isdir(path) &&
                isfile(joinpath(path, "calibration_summary.jld2")) &&
                isfile(joinpath(path, "config.jld2")),
        readdir(directory; join = true),
    )
    length(runs) >= 3 || error(
        "Expected at least three complete calibration strengths below $directory, found $(length(runs)).",
    )
    return runs
end

function calibration_evaluation_records(run_directory::AbstractString)
    directory = joinpath(run_directory, "evaluations")
    isdir(directory) || error("Calibration run has no evaluations: $run_directory")
    files = sort!(filter(
        path -> startswith(basename(path), "update_") && endswith(path, ".jld2"),
        readdir(directory; join = true),
    ))
    isempty(files) && error("Calibration run has no evaluation shards: $run_directory")

    records = Dict{Symbol, Any}[]
    for path in files
        loaded = JLD2.load(path)
        for raw_record in loaded["candidates"]
            record = Dict{Symbol, Any}(
                (key isa Symbol ? key : Symbol(key)) => value
                for (key, value) in raw_record
            )
            record[:source_run_directory] = run_directory
            push!(records, record)
        end
    end
    sort!(records; by = record -> Int(record[:update]))
    return records
end

function load_calibration_combination(protocol::Symbol, grouping::Symbol)
    available_runs = NamedTuple[]
    for run_directory in calibration_run_directories(protocol, grouping)
        loaded_config = JLD2.load(joinpath(run_directory, "config.jld2"))
        loaded_summary = JLD2.load(joinpath(run_directory, "calibration_summary.jld2"))
        config = Dict{Symbol, Any}(
            (key isa Symbol ? key : Symbol(key)) => value
            for (key, value) in loaded_config["config"]
        )
        Symbol(config[:protocol]) === protocol || error(
            "Protocol mismatch in $run_directory.",
        )
        Symbol(config[:grouping]) === grouping || error(
            "Grouping mismatch in $run_directory.",
        )
        records = calibration_evaluation_records(run_directory)
        all(record -> Symbol(record[:threshold_id]) === :native, records) || error(
            "Calibration run contains non-native threshold candidates: $run_directory",
        )
        all(record -> Symbol(record[:numeric_status]) === :ok, records) || error(
            "Calibration run contains invalid candidates: $run_directory",
        )
        all(record -> Float64(record[:validation_matching]) > 0, records) || error(
            "Logarithmic plotting requires positive validation MSE values.",
        )
        pareto_front = Dict{Symbol, Any}[
            Dict{Symbol, Any}(
                (key isa Symbol ? key : Symbol(key)) => value
                for (key, value) in raw_candidate
            )
            for raw_candidate in loaded_summary["pareto_front"]
        ]
        for candidate in pareto_front
            candidate[:source_run_directory] = run_directory
            candidate[:calibration_strength] = Float64(config[:regularization_strength])
            candidate[:regularized_updates] = Int(config[:regularized_updates])
            candidate[:regression_learning_rate] = Float64(get(
                config,
                :regression_learning_rate,
                protocol === :fixed ? 1e-4 : 2e-4,
            ))
        end
        regression_learning_rate = Float64(get(
            config,
            :regression_learning_rate,
            protocol === :fixed ? 1e-4 : 2e-4,
        ))
        calibration_phase = Symbol(get(config, :calibration_phase, :baseline))
        push!(available_runs, (
            strength = Float64(config[:regularization_strength]),
            regularized_updates = Int(config[:regularized_updates]),
            regression_learning_rate,
            calibration_phase,
            run_directory,
            records,
            pareto_front,
            config,
        ))
    end
    key = (protocol, grouping)
    function complete_block(expected_strengths, expected_updates)
        block = filter(
            run -> run.regularized_updates == expected_updates &&
                   run.regression_learning_rate == P6_CALIBRATION_REGRESSION_LEARNING_RATE &&
                   run.calibration_phase === P6_CALIBRATION_PHASE &&
                   run.strength in expected_strengths,
            available_runs,
        )
        return length(block) == length(expected_strengths) &&
               sort([run.strength for run in block]) == sort(expected_strengths) ? block : NamedTuple[]
    end

    runs = complete_block(
        P6_CALIBRATION_STRENGTHS[key],
        P6_CALIBRATION_UPDATES[protocol],
    )
    study_complete = !isempty(runs)
    study_complete || error(
        "$protocol/$grouping has no complete $(P6_CALIBRATION_PHASE) block with " *
        "$(P6_CALIBRATION_UPDATES[protocol]) updates and regression learning rate " *
        "$P6_CALIBRATION_REGRESSION_LEARNING_RATE.",
    )
    sort!(runs; by = run -> run.strength)
    budgets = sort!(unique(run.regularized_updates for run in runs))
    return (; runs, budgets, study_complete)
end

function calibration_dominates(left, right)
    left_groups = Int(left[:active_groups])
    right_groups = Int(right[:active_groups])
    left_matching = Float64(left[:validation_matching])
    right_matching = Float64(right[:validation_matching])
    return left_groups <= right_groups &&
           left_matching <= right_matching &&
           (left_groups < right_groups || left_matching < right_matching)
end

function pooled_calibration_front(runs)
    best_by_group = Dict{Int, Dict{Symbol, Any}}()
    for run in runs, record in run.records
        groups = Int(record[:active_groups])
        candidate = copy(record)
        candidate[:calibration_strength] = run.strength
        candidate[:regularized_updates] = run.regularized_updates
        candidate[:regression_learning_rate] = run.regression_learning_rate
        if !haskey(best_by_group, groups) ||
           Float64(candidate[:validation_matching]) <
           Float64(best_by_group[groups][:validation_matching])
            best_by_group[groups] = candidate
        end
    end
    candidates = collect(values(best_by_group))
    front = filter(
        candidate -> !any(
            other -> other !== candidate && calibration_dominates(other, candidate),
            candidates,
        ),
        candidates,
    )
    sort!(front; by = record -> -Int(record[:active_groups]))
    return front
end

function calibration_protocol_label(protocol::Symbol)
    return protocol === :fixed ? "Fixed IC" : "Varying IC"
end

function calibration_grouping_label(grouping::Symbol)
    return grouping === :grouped_channels ? "grouped channels" : "separate channels"
end

function calibration_strength_label(strength::Real)
    return @sprintf("λ = %.6g", Float64(strength))
end

function calibration_strength_color(strength::Real, runs)
    strengths = sort!(unique(run.strength for run in runs))
    index = findfirst(==(Float64(strength)), strengths)
    isnothing(index) && error("Unknown calibration strength $strength.")
    index <= length(P6_CALIBRATION_STRENGTH_COLORS) || error(
        "The calibration color palette is too short for $(length(strengths)) strengths.",
    )
    return P6_CALIBRATION_STRENGTH_COLORS[index]
end

function calibration_hover_customdata(records)
    return hcat(
        Int[record[:update] for record in records],
        Int[record[:active_inputs] for record in records],
        Int[record[:active_sensor_locations] for record in records],
        Int[record[:regularized_updates] for record in records],
        Float64[record[:regression_learning_rate] for record in records],
    )
end

function calibration_hover_template(prefix::AbstractString)
    return prefix * "<br>" *
           "Update: %{customdata[0]}<br>" *
           "Active groups: %{x}<br>" *
           "Active global inputs: %{customdata[1]}<br>" *
           "Active sensor locations: %{customdata[2]}<br>" *
           "Run budget: %{customdata[3]} updates<br>" *
           "Regression LR: %{customdata[4]:.2e}<br>" *
           "Validation MSE: %{y:.6e}<extra></extra>"
end

function plot_calibration_combination(
    protocol::Symbol,
    grouping::Symbol,
    runs,
    regularized_update_budgets,
    output_directory::AbstractString,
)
    traces = PlotlyJS.GenericTrace[]
    pooled_front = pooled_calibration_front(runs)

    for run in runs
        for record in run.records
            record[:regularized_updates] = run.regularized_updates
            record[:regression_learning_rate] = run.regression_learning_rate
        end
        color = calibration_strength_color(run.strength, runs)
        label = calibration_strength_label(run.strength)
        push!(
            traces,
            scatter(
                x = Int[record[:active_groups] for record in run.records],
                y = Float64[record[:validation_matching] for record in run.records],
                mode = "markers",
                name = label,
                legendgroup = label,
                customdata = calibration_hover_customdata(run.records),
                marker = attr(
                    color = color,
                    size = 7,
                    opacity = 0.58,
                    symbol = "circle",
                    line = attr(color = "white", width = 0.5),
                ),
                hovertemplate = calibration_hover_template(label),
            ),
        )

        strength_front = filter(
            record -> Float64(record[:calibration_strength]) == run.strength,
            pooled_front,
        )
        isempty(strength_front) && continue
        push!(
            traces,
            scatter(
                x = Int[record[:active_groups] for record in strength_front],
                y = Float64[record[:validation_matching] for record in strength_front],
                mode = "markers",
                name = "$label — pooled Pareto",
                legendgroup = label,
                customdata = calibration_hover_customdata(strength_front),
                marker = attr(
                    color = color,
                    size = 13,
                    opacity = 1.0,
                    symbol = "diamond",
                    line = attr(color = "#111111", width = 1.8),
                ),
                hovertemplate = calibration_hover_template("$label — pooled Pareto"),
            ),
        )
    end

    title = "$(calibration_protocol_label(protocol)) — " *
            "$(calibration_grouping_label(grouping)) — " *
            join(format_calibration_integer.(regularized_update_budgets), " + ") *
            " update runs"
    plot_handle = Plot(
        traces,
        Layout(
            template = "plotly_white",
            title = attr(
                text = title,
                x = 0.5,
                xanchor = "center",
                font = attr(size = 22, color = "#252525"),
            ),
            paper_bgcolor = "white",
            plot_bgcolor = "white",
            width = 900,
            height = 560,
            margin = attr(l = 100, r = 30, t = 80, b = 80),
            font = attr(family = "Arial, sans-serif", size = 15, color = "#303030"),
            xaxis = attr(
                title = attr(text = "Active groups (lower is sparser)", standoff = 12),
                showline = true,
                mirror = true,
                linecolor = "#3A3A3A",
                linewidth = 1,
                ticks = "outside",
                gridcolor = "#E6E6E6",
                zeroline = false,
                dtick = grouping === :grouped_channels ? 2 : 5,
            ),
            yaxis = attr(
                title = attr(
                    text = "Autoregressive validation MSE (log scale)",
                    standoff = 12,
                ),
                type = "log",
                showline = true,
                mirror = true,
                linecolor = "#3A3A3A",
                linewidth = 1,
                ticks = "outside",
                gridcolor = "#E6E6E6",
                zeroline = false,
            ),
            legend = attr(
                x = 0.985,
                y = 0.985,
                xanchor = "right",
                yanchor = "top",
                bgcolor = "rgba(255, 255, 255, 0.92)",
                bordercolor = "#CFCFCF",
                borderwidth = 1,
                font = attr(size = 12),
            ),
            hovermode = "closest",
        ),
    )

    mkpath(output_directory)
    stem = "$(protocol)_$(grouping)_pareto"
    output_path = joinpath(output_directory, "$stem.svg")
    PlotlyJS.savefig(plot_handle, output_path; width = 900, height = 560)
    return (; plot_handle, output_path, pooled_front)
end

function format_calibration_integer(value::Integer)
    value = Int(value)
    value < 1_000 && return string(value)
    return "$(value ÷ 1_000),$(lpad(string(value % 1_000), 3, '0'))"
end

function calibration_test_worker_command(combination)
    julia = Base.julia_cmd()
    project_root = normpath(joinpath(@__DIR__, "..", ".."))
    return `$julia --startup-file=no --project=$project_root $P6_CALIBRATION_TEST_WORKER --protocol $(string(combination.protocol)) --grouping $(string(combination.grouping))`
end

function require_complete_calibration_test_blocks!()
    for combination in P6_CALIBRATION_PLOT_COMBINATIONS
        loaded = load_calibration_combination(
            combination.protocol,
            combination.grouping,
        )
        loaded.study_complete || error(
            "Closed-loop test diagnostics require the complete long-budget calibration " *
            "block for $(combination.protocol)/$(combination.grouping): expected " *
            "$(P6_CALIBRATION_UPDATES[combination.protocol]) updates for all frozen strengths. " *
            "No test episode has been started.",
        )
    end
    return nothing
end

"""
    inspect_strength_calibration_pilot(; output_directory=...)

Generate one color-coded, point-only Pareto plot for every Fixed/Varying and
grouped/separate calibration combination. Small circles show all validation
checkpoints; larger diamonds show the pooled nondominated checkpoints. By
default, all retained per-strength Pareto candidates are then evaluated in
closed loop and compared with the expert. Set `closed_loop_test=false` to
generate only the validation plots without consuming the Varying test split.
"""
function inspect_strength_calibration_pilot(
    ;
    output_directory::AbstractString = P6_CALIBRATION_ANALYSIS_DIRECTORY,
    closed_loop_test::Bool = true,
)
    outputs = Dict{Tuple{Symbol, Symbol}, Any}()
    for combination in P6_CALIBRATION_PLOT_COMBINATIONS
        loaded = load_calibration_combination(
            combination.protocol,
            combination.grouping,
        )
        result = plot_calibration_combination(
            combination.protocol,
            combination.grouping,
            loaded.runs,
            loaded.budgets,
            output_directory,
        )
        outputs[(combination.protocol, combination.grouping)] = result
        println(result.output_path)
    end
    if closed_loop_test
        require_complete_calibration_test_blocks!()
        println()
        println("Running cached closed-loop test diagnostics in fresh Julia processes...")
        for combination in P6_CALIBRATION_PLOT_COMBINATIONS
            println(
                "  $(calibration_protocol_label(combination.protocol)), " *
                calibration_grouping_label(combination.grouping),
            )
            run(calibration_test_worker_command(combination))
        end
    end
    return outputs
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        inspect_strength_calibration_pilot()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
