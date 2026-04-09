
#save_number = 1001




function start()
    # include("./GrOWL/MAT_expert_apprentice.jl")
    # randomIC = false
    # Base.@invokelatest load_apprentice()

    # include("./GrOWL/train_masked/masked_fixedIC.jl")

    # for i in 1:5
    #     Base.@invokelatest train()

    #     Base.@invokelatest save(save_number)
    # end

    println("Now with random IC")

    include("./GrOWL/MAT_expert_apprentice.jl")
    randomIC = true
    Base.@invokelatest load_apprentice()

    include("./GrOWL/train_masked/masked_rIC.jl")

    for i in 1:5
        Base.@invokelatest train()

        Base.@invokelatest save(save_number)
    end
end

