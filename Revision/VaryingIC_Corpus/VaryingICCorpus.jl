using Dates
using JLD2
using Oceananigans
using Printf
using Random
using StableRNGs
import Plots

const SCHEMA_VERSION = 1
const VALID_SPLITS = (:train, :validation, :test)
const DEFAULT_SPLIT_SIZES = Dict(:train => 20, :validation => 1, :test => 2)

const NX = 96
const NZ = 64
const LX = 2π
const LZ = 2.0
const RA = 1.0e4
const PR = 0.7
const DELTA_B = 1.0
const INITIAL_PERTURBATION = 0.2
const INNER_DT = 0.03
const SPINUP_TIME = 300.0

const CORPUS_PATH = joinpath(@__DIR__, "varying_ic_corpus.jld2")
const PLOT_DIRECTORY = joinpath(@__DIR__, "plots")

const SIMULATION_CONFIG = Dict{Symbol, Any}(
    :Nx => NX,
    :Nz => NZ,
    :Lx => LX,
    :Lz => LZ,
    :Ra => RA,
    :Pr => PR,
    :delta_b => DELTA_B,
    :initial_perturbation => INITIAL_PERTURBATION,
    :inner_dt => INNER_DT,
    :spinup_time => SPINUP_TIME,
    :advection => "UpwindBiasedFifthOrder",
    :timestepper => "RungeKutta3",
    :top_b => 1.0,
    :bottom_b => 2.0,
)

function empty_corpus()
    corpus = Dict{Symbol, Dict{Int, Any}}()
    for split in VALID_SPLITS
        corpus[split] = Dict{Int, Any}()
    end
    return corpus
end

function normalize_split(split)::Symbol
    split_symbol = Symbol(lowercase(string(split)))
    split_symbol = get(
        Dict(
            :training => :train,
            :val => :validation,
            :valid => :validation,
            :testing => :test,
        ),
        split_symbol,
        split_symbol,
    )
    split_symbol in VALID_SPLITS || throw(
        ArgumentError("Unknown split '$split'. Use :train, :validation, or :test."),
    )
    return split_symbol
end

function normalize_loaded_corpus(raw_corpus)
    corpus = empty_corpus()
    for split in VALID_SPLITS
        raw_split = if haskey(raw_corpus, split)
            raw_corpus[split]
        elseif haskey(raw_corpus, string(split))
            raw_corpus[string(split)]
        else
            nothing
        end

        isnothing(raw_split) && continue
        for (seed, snapshot) in raw_split
            corpus[split][Int(seed)] = snapshot
        end
    end
    return corpus
end

"""
    load_corpus(path=CORPUS_PATH)

Load the persisted split dictionary from a JLD2 file.
If the file does not exist, return an empty dictionary with train, validation,
and test entries.
"""
function load_corpus(path::AbstractString = CORPUS_PATH)
    isfile(path) || return empty_corpus()
    loaded = JLD2.load(path)
    haskey(loaded, "corpus") || error("JLD2 corpus file '$path' has no 'corpus' entry.")
    return normalize_loaded_corpus(loaded["corpus"])
end

const CORPUS = load_corpus()

"""
    save_corpus!(; corpus=CORPUS, path=CORPUS_PATH)

Persist the complete corpus and its simulation metadata as JLD2.
"""
function save_corpus!(; corpus = CORPUS, path::AbstractString = CORPUS_PATH)
    mkpath(dirname(path))
    JLD2.jldsave(
        path;
        corpus = corpus,
        schema_version = SCHEMA_VERSION,
        simulation_config = SIMULATION_CONFIG,
    )
    return path
end

"""
    reload_corpus!(path=CORPUS_PATH)

Replace the contents of the global in-memory corpus with the current JLD2 file.
"""
function reload_corpus!(path::AbstractString = CORPUS_PATH)
    loaded = load_corpus(path)
    empty!(CORPUS)
    for split in VALID_SPLITS
        CORPUS[split] = loaded[split]
    end
    return CORPUS
end

function build_spinup_model()
    grid = RectilinearGrid(
        size = (NX, NZ),
        x = (0, LX),
        z = (0, LZ),
        topology = (Periodic, Flat, Bounded),
    )

    viscosity = sqrt(PR / RA)
    diffusivity = 1 / sqrt(PR * RA)

    u_bcs = FieldBoundaryConditions(
        top = ValueBoundaryCondition(0),
        bottom = ValueBoundaryCondition(0),
    )
    b_bcs = FieldBoundaryConditions(
        top = ValueBoundaryCondition(1),
        bottom = ValueBoundaryCondition(2),
    )

    return NonhydrostaticModel(
        ;
        grid,
        advection = UpwindBiasedFifthOrder(),
        timestepper = :RungeKutta3,
        tracers = (:b),
        buoyancy = Buoyancy(model = BuoyancyTracer()),
        closure = ScalarDiffusivity(ν = viscosity, κ = diffusivity),
        boundary_conditions = (u = u_bcs, b = b_bcs),
        coriolis = nothing,
    )
end

function set_seeded_initial_conditions!(model, seed::Integer)
    rng = StableRNG(Int(seed))
    u_shape = size(interior(model.velocities.u))
    w_shape = size(interior(model.velocities.w))
    b_shape = size(interior(model.tracers.b))

    u_initial = INITIAL_PERTURBATION .* randn(rng, Float64, u_shape)
    w_initial = INITIAL_PERTURBATION .* randn(rng, Float64, w_shape)
    b_noise = INITIAL_PERTURBATION .* randn(rng, Float64, b_shape)

    z_centers = collect(range(LZ / (2NZ), step = LZ / NZ, length = NZ))
    b_background = reshape(1 .+ (2 .- z_centers) .* DELTA_B ./ 2, 1, 1, NZ)
    b_initial = b_background .+ b_noise

    set!(model, u = u_initial, w = w_initial, b = b_initial)
    return model
end

function copy_interior(field)
    return Float32.(Array(interior(field)))
end

"""
    generate_basis_snapshot(seed; show_progress=true)

Generate one independently seeded RBC basis state by spinning the configuration
used in `randomIC/randomIC_MAT.jl` from noisy initial fields to time 300.
No file is written by this low-level function.
"""
function generate_basis_snapshot(seed::Integer; show_progress::Bool = true)
    model = build_spinup_model()
    set_seeded_initial_conditions!(model, seed)

    simulation = Simulation(model, Δt = INNER_DT, stop_time = SPINUP_TIME)
    simulation.verbose = false

    if show_progress
        start_time = time_ns()
        function report_progress(sim)
            @printf(
                "basis seed %d | iteration %d | simulation time %.2f / %.2f | wall time %s\n",
                seed,
                sim.model.clock.iteration,
                sim.model.clock.time,
                SPINUP_TIME,
                prettytime(1e-9 * (time_ns() - start_time)),
            )
        end
        simulation.callbacks[:corpus_progress] = Callback(
            report_progress,
            IterationInterval(1000),
        )
    end

    run!(simulation)

    return Dict{Symbol, Any}(
        :seed => Int(seed),
        :u => copy_interior(model.velocities.u),
        :w => copy_interior(model.velocities.w),
        :b => copy_interior(model.tracers.b),
        :simulation_time => Float64(model.clock.time),
        :iteration => Int(model.clock.iteration),
        :created_at => string(Dates.now()),
        :schema_version => SCHEMA_VERSION,
    )
end

function seed_locations(corpus, seed::Integer)
    return [split for split in VALID_SPLITS if haskey(corpus[split], Int(seed))]
end

function draw_unused_seed(rng::AbstractRNG, corpus)
    for _ in 1:10_000
        seed = rand(rng, 1:typemax(Int32))
        isempty(seed_locations(corpus, seed)) && return seed
    end
    error("Could not draw an unused seed after 10,000 attempts.")
end

"""
    add_basis_snapshot!(split, seed; ...)

Generate a basis snapshot and insert it into an existing split dictionary under
its seed.
The corpus is saved after insertion and overview plots are refreshed by default.
"""
function add_basis_snapshot!(
    split,
    seed::Integer;
    corpus = CORPUS,
    overwrite::Bool = false,
    save_after::Bool = true,
    refresh_plots::Bool = true,
    show_progress::Bool = true,
)
    split_symbol = normalize_split(split)
    seed = Int(seed)
    locations = seed_locations(corpus, seed)

    if !isempty(locations)
        if locations != [split_symbol]
            error("Seed $seed already belongs to split(s) $(locations) and cannot be reused.")
        elseif !overwrite
            error("Seed $seed already exists in split :$split_symbol. Pass overwrite=true to replace it.")
        end
    end

    snapshot = generate_basis_snapshot(seed; show_progress = show_progress)
    corpus[split_symbol][seed] = snapshot
    save_after && save_corpus!(; corpus)
    refresh_plots && plot_corpus_overviews(; corpus)
    return snapshot
end

"""
    generate_to_size!(split, target_count; rng=Random.default_rng(), ...)

Fill a split up to `target_count` with newly drawn unique seeds.
Existing entries are preserved.
"""
function generate_to_size!(
    split,
    target_count::Integer;
    corpus = CORPUS,
    rng::AbstractRNG = Random.default_rng(),
    save_after_each::Bool = true,
    refresh_plots::Bool = true,
    show_progress::Bool = true,
)
    target_count >= 0 || throw(ArgumentError("target_count must be nonnegative."))
    split_symbol = normalize_split(split)
    generated_seeds = Int[]

    while length(corpus[split_symbol]) < target_count
        seed = draw_unused_seed(rng, corpus)
        add_basis_snapshot!(
            split_symbol,
            seed;
            corpus,
            save_after = save_after_each,
            refresh_plots = false,
            show_progress,
        )
        push!(generated_seeds, seed)
    end

    !save_after_each && !isempty(generated_seeds) && save_corpus!(; corpus)
    refresh_plots && plot_corpus_overviews(; corpus)
    return generated_seeds
end

"""
    generate_default_corpus!(; rng=Random.default_rng(), ...)

Fill the corpus to 20 training, one validation, and two test basis snapshots.
Nothing is generated merely by including this file.
"""
function generate_default_corpus!(
    ;
    corpus = CORPUS,
    rng::AbstractRNG = Random.default_rng(),
    show_progress::Bool = true,
)
    generated = Dict{Symbol, Vector{Int}}()
    for split in VALID_SPLITS
        generated[split] = generate_to_size!(
            split,
            DEFAULT_SPLIT_SIZES[split];
            corpus,
            rng,
            save_after_each = true,
            refresh_plots = false,
            show_progress,
        )
    end
    plot_corpus_overviews(; corpus)
    return generated
end

"""
    delete_basis_snapshot!(split, seed; ...)

Delete one basis snapshot, persist the updated dictionary, and refresh plots.
The next `generate_default_corpus!` call will refill a split that is below its
default target size with a newly drawn seed.
"""
function delete_basis_snapshot!(
    split,
    seed::Integer;
    corpus = CORPUS,
    save_after::Bool = true,
    refresh_plots::Bool = true,
)
    split_symbol = normalize_split(split)
    removed = pop!(corpus[split_symbol], Int(seed), nothing)
    isnothing(removed) && return nothing
    save_after && save_corpus!(; corpus)
    refresh_plots && plot_corpus_overviews(; corpus)
    return removed
end

function snapshot_value(snapshot, key::Symbol)
    if haskey(snapshot, key)
        return snapshot[key]
    elseif haskey(snapshot, string(key))
        return snapshot[string(key)]
    end
    error("Snapshot has no '$key' entry.")
end

function horizontal_shift(field, offset::Integer)
    shifts = ntuple(dimension -> dimension == 1 ? Int(offset) : 0, ndims(field))
    return circshift(field, shifts)
end

function transformed_field(snapshot, key::Symbol, mirror::Bool, offset::Int)
    field = copy(snapshot_value(snapshot, key))
    if mirror
        field = reverse(field; dims = 1)
        if key === :u
            field = horizontal_shift(field, 1)
            field .*= -1
        end
    end
    return horizontal_shift(field, offset)
end

"""
    sample_initial_condition(
        split;
        rng=Random.default_rng(),
        mirror=nothing,
        offset=nothing,
        corpus=CORPUS,
    )

Randomly choose a basis snapshot from a split, optionally mirror it horizontally,
and apply a periodic horizontal offset.
If `mirror` or `offset` is `nothing`, that choice is sampled from `rng`.
Offsets are normalized to `0:NX-1`.

Horizontal reflection reverses the first array dimension.
For the face-centered horizontal velocity `u`, it additionally applies the
one-index periodic alignment shift required by the staggered grid and changes
the velocity sign.
"""
function sample_initial_condition(
    split;
    rng::AbstractRNG = Random.default_rng(),
    mirror::Union{Nothing, Bool} = nothing,
    offset::Union{Nothing, Integer} = nothing,
    corpus = CORPUS,
)
    split_symbol = normalize_split(split)
    split_corpus = corpus[split_symbol]
    isempty(split_corpus) && error("Split :$split_symbol contains no basis snapshots.")

    available_seeds = sort!(collect(keys(split_corpus)))
    base_seed = rand(rng, available_seeds)
    mirror_used = isnothing(mirror) ? rand(rng, Bool) : mirror
    offset_used = isnothing(offset) ? rand(rng, 0:(NX - 1)) : mod(Int(offset), NX)
    snapshot = split_corpus[base_seed]

    return (
        u = transformed_field(snapshot, :u, mirror_used, offset_used),
        w = transformed_field(snapshot, :w, mirror_used, offset_used),
        b = transformed_field(snapshot, :b, mirror_used, offset_used),
        split = split_symbol,
        base_seed = base_seed,
        mirror = mirror_used,
        offset = offset_used,
    )
end

function field_matrix(snapshot, field::Symbol)
    values = snapshot_value(snapshot, field)
    if ndims(values) == 2
        return Array(values)
    elseif ndims(values) == 3 && size(values, 2) == 1
        return dropdims(Array(values); dims = 2)
    end
    error("Expected a two-dimensional field or a three-dimensional field with a flat second dimension, got size $(size(values)).")
end

function field_limits(corpus, field::Symbol)
    limits = Float64[]
    for split in VALID_SPLITS
        for snapshot in values(corpus[split])
            matrix = field_matrix(snapshot, field)
            push!(limits, minimum(matrix), maximum(matrix))
        end
    end
    isempty(limits) && return nothing
    return extrema(limits)
end

function plot_split_overview(
    split::Symbol;
    corpus = CORPUS,
    field::Symbol = :b,
    output_directory::AbstractString = PLOT_DIRECTORY,
    color_limits = field_limits(corpus, field),
)
    split_corpus = corpus[split]
    seeds = sort!(collect(keys(split_corpus)))
    mkpath(output_directory)

    output_name = Dict(
        :train => "training_snapshots.png",
        :validation => "validation_snapshots.png",
        :test => "test_snapshots.png",
    )[split]
    output_path = joinpath(output_directory, output_name)

    if isempty(seeds)
        overview = Plots.plot(
            title = "$(uppercasefirst(string(split))) set: no snapshots",
            axis = false,
            grid = false,
            legend = false,
            size = (500, 250),
        )
        Plots.savefig(overview, output_path)
        return output_path
    end

    panels = [
        Plots.heatmap(
            field_matrix(split_corpus[seed], field)';
            title = "seed $seed",
            titlefontsize = 8,
            color = :thermal,
            clims = color_limits,
            colorbar = false,
            axis = false,
            ticks = false,
            framestyle = :box,
        )
        for seed in seeds
    ]

    columns = min(5, length(panels))
    rows = cld(length(panels), 5)
    overview = Plots.plot(
        panels...;
        layout = (rows, columns),
        size = (260 * columns, 190 * rows),
        plot_title = "$(uppercasefirst(string(split))) basis snapshots ($(field))",
        legend = false,
    )
    Plots.savefig(overview, output_path)
    return output_path
end

"""
    plot_corpus_overviews(; corpus=CORPUS, field=:b, output_directory=PLOT_DIRECTORY)

Write one PNG overview per split.
Each mini-plot shows one basis field, with at most five snapshots per row.
The default overview field is buoyancy/temperature `b`.
"""
function plot_corpus_overviews(
    ;
    corpus = CORPUS,
    field::Symbol = :b,
    output_directory::AbstractString = PLOT_DIRECTORY,
)
    Plots.gr()
    color_limits = field_limits(corpus, field)
    return Dict(
        split => plot_split_overview(
            split;
            corpus,
            field,
            output_directory,
            color_limits,
        )
        for split in VALID_SPLITS
    )
end
