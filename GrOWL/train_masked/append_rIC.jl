
#save_number = 1010




function start_append()

    include("./GrOWL/MAT_expert_apprentice.jl")
    randomIC = true
    Base.@invokelatest load_apprentice()

    include("./GrOWL/train_masked/masked_rIC.jl")

    Base.@invokelatest load(save_number)

    for i in 1:5
        Base.@invokelatest train()

        Base.@invokelatest save(save_number)
    end

end

start_append()