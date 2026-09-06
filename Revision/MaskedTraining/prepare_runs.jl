include("MaskedStudy.jl")
using .MaskedStudy, JLD2
const S = MaskedStudy

function main(args = ARGS)
    defaults = Dict{String, Any}("results_dir" => S.DEFAULT_RESULTS, "comparison_dir" => S.DEFAULT_COMPARISON,
        "package7_results" => joinpath(S.ROOT, "Revision/Package7/results/260830_173924"),
        "package8_results" => joinpath(S.ROOT, "Revision/Package8/results/260830_231109"),
        "protocol" => "all", "grouping" => "all", "jobs_file" => nothing, "preview" => false)
    o = S.parse_options(args, defaults; flags = ["preview"])
    o["protocol"] in ("all", "fixed", "varying") || error("Invalid protocol")
    o["grouping"] in ("all", "gc", "sc") || error("Invalid grouping")
    m = S.build_manifest(; comparison = abspath(o["comparison_dir"]),
        package7 = abspath(o["package7_results"]), package8 = abspath(o["package8_results"]))
    path = S.manifest_path(o["results_dir"])
    if isfile(path)
        JLD2.load(path, "manifest").identity == m.identity || error("Frozen manifest changed; use a new --results-dir.")
    elseif !o["preview"]
        S.atomic_save(path; manifest = m)
    end
    rows = String["protocol\tgrouping\trun_id"]
    for c in m.candidates
        println("$(c.protocol)/$(c.grouping): $(c.configuration), $(c.active_groups)/$(c.grouping === :gc ? 32 : 96), MSE=$(c.validation_mse)")
        o["protocol"] in ("all", string(c.protocol)) || continue
        o["grouping"] in ("all", string(c.grouping)) || continue
        for e in m.entries
            target = S.result_path(o["results_dir"], c.protocol, c.grouping, e.run_id)
            S.complete_result(target, m.identity, S.BUDGETS[c.protocol]) && continue
            push!(rows, "$(c.protocol)\t$(c.grouping)\t$(e.run_id)")
        end
    end
    isnothing(o["jobs_file"]) || write(o["jobs_file"], join(rows, "\n") * "\n")
    println("$(length(rows)-1) pending training workers; Ra=$(m.rayleigh).")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
