
#save_number = 1010




function start_append()
    include("./RB_Joon/RB_Joon_PE_fixed_ic_MAT.jl")

    Base.@invokelatest load(save_number)

    Base.@invokelatest train()

    Base.@invokelatest save(save_number)





    # include("./RB_Joon/RB_Joon_PE_fixed_ic_PPO.jl")

    # Base.@invokelatest load(save_number)

    # Base.@invokelatest train()

    # Base.@invokelatest save(save_number)
end

