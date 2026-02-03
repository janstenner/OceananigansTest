
save_number = 1000




function start()
    include("./randomIC_MAT.jl")

    train()

    save(save_number)





    include("./randomIC_PPO.jl")

    train()

    save(save_number)
end

