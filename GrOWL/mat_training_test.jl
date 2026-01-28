rand_inds = shuffle!(rng, Vector(1:100))

# if !(@isdefined states)
#     generate_states()
# end

temp_batch_size = 1

#batch = states[:, :, rand_inds[1:temp_batch_size]]

na = size(apprentice.decoder.embedding.weight)[2]
na = 1

batch = env.state
# μ_expert = agent.policy.approximator.actor(batch)[1]
# μ_expert = agent(env)
# RL.test_update(apprentice, batch, μ_expert)

global g_decoder
global mu

g_decoder = Flux.gradient(apprentice.decoder) do p_decoder

    obsrep, val = apprentice.encoder(batch)

    μ_expert = prob(agent.policy, batch, nothing).μ

    temp_act = cat(zeros(Float32,na,1,temp_batch_size),μ_expert[:,1:end-1,:],dims=2)

    μ, logσ = p_decoder(temp_act, obsrep[:,:,:]) # Zeros do not work here


    diff = μ - μ_expert
    mse = mean(diff.^2)

    Zygote.@ignore println(mse)

   return mse
end

Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder[1])








# function test_update(apprentice::MATPolicy, batch, μ_expert)
    
#     na = size(apprentice.decoder.embedding.weight)[2]
    
#     g_decoder = Flux.gradient(apprentice.decoder) do p_decoder

#         obsrep, val = apprentice.encoder(batch)

#         temp_act = cat(zeros(Float32,1,1,1),μ_expert[:,1:end-1,:],dims=2)

#         μ, logσ = p_decoder(temp_act, obsrep[:,:,:]) # Zeros do not work here


#         diff = μ - μ_expert
#         mse = mean(diff.^2)

#         Zygote.@ignore println(mse)

#         return mse
#     end

#     Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder[1])

# end