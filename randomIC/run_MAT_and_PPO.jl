
save_number = 1000




function start()
    include("./randomIC/randomIC_MAT.jl")

    Base.@invokelatest train()

    Base.@invokelatest save(save_number)





    include("./randomIC/randomIC_PPO.jl")

    Base.@invokelatest train()

    Base.@invokelatest save(save_number)
end

