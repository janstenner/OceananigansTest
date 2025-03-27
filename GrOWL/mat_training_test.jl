rand_inds = shuffle!(rng, Vector(1:100))

if !(@isdefined states)
    generate_states()
end

temp_batch_size = 1

batch = states[:, :, rand_inds[1:temp_batch_size]]

na = size(apprentice.decoder.embedding.weight)[2]

global g_decoder

g_decoder = Flux.gradient(apprentice.decoder) do p_decoder

    obsrep, val = apprentice.encoder(batch)

    μ_expert = agent.policy.approximator.actor(batch)[1]

    temp_act = cat(zeros(Float32,na,1,temp_batch_size),μ_expert[:,1:end-1,:],dims=2)

    μ, logσ = p_decoder(temp_act, obsrep[:,:,:]) # Zeros do not work here


    diff = μ - μ_expert
    mse = mean(diff.^2)

    Zygote.@ignore println(mse)

    return mse
end

Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder[1])