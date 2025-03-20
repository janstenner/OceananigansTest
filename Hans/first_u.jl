using Zygote
using Optimisers
using Flux



Base.@kwdef struct u_mpc
    n_base_functions_x = 13
    n_base_functions_t = 13
    c::Matrix{Float32} = randn(n_base_functions_x,n_base_functions_t) .* 0.2
    base_functions_space = [[x -> 1]..., [x -> sin(x*i*2π/Lx) for i in 1:Int((n_base_functions_x-1)/2)]..., [x -> cos(x*i*2π/Lx) for i in 1:Int((n_base_functions_x-1)/2)]...]
    base_functions_time = [[t -> 1]..., [t -> sin(t*i*2π/(te/2)) for i in 1:Int((n_base_functions_t-1)/2)]..., [t -> cos(t*i*2π/(te/2)) for i in 1:Int((n_base_functions_t-1)/2)]...]
end

Flux.@functor u_mpc

function (st::u_mpc)(x,t)
    sum(st.c[i,j] * st.base_functions_space[i](x) * st.base_functions_time[j](t) for i in 1:st.n_base_functions_x, j in 1:st.n_base_functions_t)
end

function (st::u_mpc)(x::AbstractVector, t::AbstractVector)
    # Evaluate each basis function on all x and t values.
    S = [f.(x) for f in st.base_functions_space]  # Each S[i] is a vector with length(x)
    T = [g.(t) for g in st.base_functions_time]     # Each T[j] is a vector with length(t)
    
    # Initialize the result array.
    result = zeros(Float32, length(x), length(t))
    
    @inbounds for i in 1:st.n_base_functions_x
        for j in 1:st.n_base_functions_t
            # Compute the outer product: S[i] is treated as a column vector and T[j]' as a row vector.
            # This produces a matrix where each element is S[i][p] * T[j][q].
            result += st.c[i, j] * (S[i] * T[j]')
        end
    end
    return result
end

load()
agent.policy.approximator.actor.logσ[1] = -14.0f0

total_steps = 4_000

u = u_mpc()

new_learning_rate = 1e-1
optimizer_u = Optimisers.Adam(new_learning_rate, betas)
u_state_tree = Flux.setup(optimizer_u, u)



function check_collected()
    global states
    global collected_actions

    if !(@isdefined states) || !(@isdefined collected_actions)
        states = zeros(Float32, size(env.state)[1], size(env.state)[2], 100)
        collected_actions = zeros(Float32, actuators, 100)

        reset!(env)
        for i in 1:100

            action = agent(env)

            states[:, :, i] .= env.state
            collected_actions[:, i] .= action[:]

            env(action)

            println(i, "% of simulation done")
        end
    end
end


function u_mpc_train(total_steps = 100)

    check_collected()

    global states
    global collected_actions


    global losses = Float32[]

    global xx = collect(Lx/actuators:Lx/actuators:Lx)
    global tt = collect(te/200:te/200:te/2)


    for i in 1:total_steps
        
        g_u = Flux.gradient(u) do u_p

            diff = u_p(xx,tt) - collected_actions
            mse = mean(diff.^2)

            Zygote.@ignore push!(losses, mse)

            return mse
        end

        Flux.update!(u_state_tree, u, g_u[1])

        i%10 == 0 && println(i*100/total_steps, "% done")
        println(losses[end])
    end

    plot(losses)
end


function compare_u()
    check_collected()

    global collected_actions
    global collected_u = zeros(size(collected_actions))

    global xx = collect(Lx/actuators:Lx/actuators:Lx)
    global tt = collect(te/200:te/200:te/2)

    
    collected_u[:,:] = u(xx,tt)



    p = plot(surface(z=collected_actions))
    display(p)
    p = plot(surface(z=collected_u))
    display(p)
end


function render_run_u()
    global rewards = Float64[]
    global collected_actions_2 = zeros(200,actuators)

    global xx = collect(Lx/actuators:Lx/actuators:Lx)
    global tt = collect(te/200:te/200:te/2)
    
    reward_sum = 0.0

    rm("frames/", recursive=true, force=true)
    mkdir("frames")

    colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
    ymax = 30
    layout = Layout(
            plot_bgcolor="#f1f3f7",
            coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
        )


    reset!(env)
    generate_random_init()

    for i in 1:200

        action = u(xx, [tt[i > 96 ? 96 : i]])'

        collected_actions_2[i,:] = action[:]
        env(action)

        result = env.y[1,:,:]
        result_W = env.y[2,:,:]
        result_U = env.y[3,:,:]

        p = make_subplots(rows=1, cols=1)

        add_trace!(p, heatmap(z=result', coloraxis="coloraxis"), col = 1)
        #add_trace!(p, heatmap(z=result_W'), col = 2)
        #add_trace!(p, heatmap(z=result_U'), col = 3)

        # p = plot(heatmap(z=result', coloraxis="coloraxis"), layout)

        relayout!(p, layout.fields)

        savefig(p, "frames/a$(lpad(string(i), 4, '0')).png"; width=1600, height=800)
        #body!(w,p)

        temp_reward = reward_function(env; returnGlobalNu = true)
        println(temp_reward)

        reward_sum += temp_reward
        push!(rewards, temp_reward)

        # println(mean(env.reward))

        # reward_sum += mean(env.reward)
        # push!(rewards, mean(env.reward))
    end

    println(reward_sum)



    if true
        isdir("video_output") || mkdir("video_output")
        rm("video_output/$scriptname.mp4", force=true)
        #run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "video_output/$scriptname.mp4"`)

        run(`ffmpeg -framerate 16 -i "frames/a%04d.png" -c:v libx264 -preset slow  -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 -codec:a aac "video_output/$scriptname.mp4"`)
    end
end

