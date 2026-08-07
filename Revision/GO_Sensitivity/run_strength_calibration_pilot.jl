# Package-6 one-seed calibration over three strengths and all four
# protocol/grouping combinations. This is plain Julia and works on Windows.

const P6_CALIBRATION_PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const P6_CALIBRATION_WORKER = joinpath(@__DIR__, "run_strength_calibration_worker.jl")
const P6_CALIBRATION_COMBINATIONS = (
    (protocol = :fixed, group_channels = true),
    (protocol = :fixed, group_channels = false),
    (protocol = :varying, group_channels = true),
    (protocol = :varying, group_channels = false),
)

function calibration_process_command(combination)
    julia = Base.julia_cmd()
    return `$julia --startup-file=no --project=$P6_CALIBRATION_PROJECT_ROOT $P6_CALIBRATION_WORKER --protocol $(string(combination.protocol)) --group-channels $(string(combination.group_channels))`
end

function run_strength_calibration_pilot(; preview::Bool = false)
    println("Package-6 one-seed calibration pilot")
    println("  combinations: 4")
    println("  strengths per combination: 3")
    println("  total runs: 12")
    println("  execution: sequential fresh Julia processes")
    println()

    for (index, combination) in enumerate(P6_CALIBRATION_COMBINATIONS)
        grouping = combination.group_channels ? "grouped channels" : "separate channels"
        command = calibration_process_command(combination)
        println("[$index/4] $(combination.protocol), $grouping")
        if preview
            println("  $command")
        else
            run(command)
        end
    end
    preview || println("All 12 Package-6 calibration runs are complete.")
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        run_strength_calibration_pilot()
    catch error_value
        Base.display_error(stderr, error_value, catch_backtrace())
        exit(1)
    end
end
