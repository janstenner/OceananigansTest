rand_inds = shuffle!(rng, Vector(1:100))

if !(@isdefined states)
    generate_states()
end

batch = states[:, :, rand_inds[1:batch_size]]

na = size(apprentice.decoder.embedding.weight)[2]

global g_decoder

g_decoder = Flux.gradient(apprentice.decoder) do p_decoder

    obsrep, val = apprentice.encoder(batch)

    μ, logσ = p_decoder(zeros(Float32,na,12,batch_size), obsrep[:,:,:]) # Zeros do not work here


    diff = μ - agent.policy.approximator.actor(batch)[1]
    mse = mean(diff.^2)

    Zygote.@ignore println(mse)

    return mse
end

Flux.update!(apprentice.decoder_state_tree, apprentice.decoder, g_decoder[1])