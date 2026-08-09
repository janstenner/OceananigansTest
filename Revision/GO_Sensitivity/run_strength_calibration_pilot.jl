# Package-6 long-budget one-seed calibration over every frozen strength. This
# local fallback is plain Julia and executes one fresh process at a time.

const P6_CALIBRATION_PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const P6_CALIBRATION_WORKER = joinpath(@__DIR__, "run_strength_calibration_worker.jl")
const P6_CALIBRATION_COMBINATIONS = (
    (protocol = :fixed, group_channels = true, strengths = (0.003, 0.006, 0.01, 0.03, 0.06, 0.09)),
    (protocol = :fixed, group_channels = false, strengths = (0.0015, 0.003, 0.006, 0.01, 0.02, 0.03)),
    (protocol = :varying, group_channels = true, strengths = (0.003, 0.008, 0.025, 0.04, 0.06)),
    (protocol = :varying, group_channels = false, strengths = (0.003, 0.008, 0.025, 0.04, 0.06)),
)

function calibration_process_command(combination, strength)
    julia = Base.julia_cmd()
    return `$julia --startup-file=no --project=$P6_CALIBRATION_PROJECT_ROOT $P6_CALIBRATION_WORKER --protocol $(string(combination.protocol)) --group-channels $(string(combination.group_channels)) --strength $(string(strength))`
end

function run_strength_calibration_pilot(; preview::Bool = false)
    println("Package-6 long-budget strength-calibration rerun")
    println("  combinations: 4")
    println("  strengths: Fixed GC 6; Fixed SC 6; Varying GC 5; Varying SC 5")
    println("  runs: 22")
    println("  regression learning rate: 2e-4")
    println("  updates: Fixed 35,000; Varying 50,000")
    println("  execution: sequential fresh Julia processes")
    println()

    jobs = [
        (combination, strength)
        for combination in P6_CALIBRATION_COMBINATIONS
        for strength in combination.strengths
    ]
    for (index, (combination, strength)) in enumerate(jobs)
        grouping = combination.group_channels ? "grouped channels" : "separate channels"
        command = calibration_process_command(combination, strength)
        println("[$index/$(length(jobs))] $(combination.protocol), $grouping, strength $strength")
        if preview
            println("  $command")
        else
            run(command)
        end
    end
    preview || println("All 22 long-budget Package-6 calibration runs are complete.")
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
