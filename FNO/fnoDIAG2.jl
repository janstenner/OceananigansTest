
fno_input_timesteps = 10

ch=(3,64,64,64,64,64, 128, 3)
modes=(16,16,fno_input_timesteps,)
σ = gelu

Transform = NeuralOperators.FourierTransform
lifting = Conv((1,1,1), ch[1]=>ch[2])
mapping = Chain(OperatorKernel(ch[2] => ch[3], modes, Transform, σ; permuted = true),
                OperatorKernel(ch[3] => ch[4], modes, Transform, σ; permuted = true),
                OperatorKernel(ch[4] => ch[5], modes, Transform, σ; permuted = true),
                OperatorKernel(ch[5] => ch[6], modes, Transform, σ; permuted = true))
project = Chain(Conv((1,1,1), ch[6]=>ch[7], σ),
                Conv((1,1,1), ch[7]=>ch[8]))



#biases for conv layers
function custom_uniform(rng::AbstractRNG, fan_in, dims...)
    bound = Float32(1/ sqrt(fan_in)) # fan_in
    return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

custom_uniform(fan_in, dims...; kwargs...) = custom_uniform(Random.GLOBAL_RNG, fan_in, dims...; kwargs...)

fan_in = first(Flux.nfan(size(lifting.weight)))
lifting.bias[:] = custom_uniform(fan_in, size(lifting.bias)...)

fan_in = first(Flux.nfan(size(project.layers[1].weight)))
project.layers[1].bias[:] = custom_uniform(fan_in, size(project.layers[1].bias)...)

fan_in = first(Flux.nfan(size(project.layers[2].weight)))
project.layers[2].bias[:] = custom_uniform(fan_in, size(project.layers[2].bias)...)

fan_in = first(Flux.nfan(size(mapping.layers[1].linear.weight)))
mapping.layers[1].linear.bias[:] = custom_uniform(fan_in, size(mapping.layers[1].linear.bias)...)

fan_in = first(Flux.nfan(size(mapping.layers[2].linear.weight)))
mapping.layers[2].linear.bias[:] = custom_uniform(fan_in, size(mapping.layers[2].linear.bias)...)

fan_in = first(Flux.nfan(size(mapping.layers[3].linear.weight)))
mapping.layers[3].linear.bias[:] = custom_uniform(fan_in, size(mapping.layers[3].linear.bias)...)

fan_in = first(Flux.nfan(size(mapping.layers[4].linear.weight)))
mapping.layers[4].linear.bias[:] = custom_uniform(fan_in, size(mapping.layers[4].linear.bias)...)

scale = one(eltype(FourierTransform)) / (ch[2] * ch[3])
mapping.layers[1].conv.weight[:,:,:] = permutedims(scale * Flux.glorot_uniform(eltype(FourierTransform),ch[3],ch[2],prod(modes)), (3,2,1))

scale = one(eltype(FourierTransform)) / (ch[3] * ch[4])
mapping.layers[2].conv.weight[:,:,:] = permutedims(scale * Flux.glorot_uniform(eltype(FourierTransform),ch[4],ch[3],prod(modes)), (3,2,1))

scale = one(eltype(FourierTransform)) / (ch[4] * ch[5])
mapping.layers[3].conv.weight[:,:,:] = permutedims(scale * Flux.glorot_uniform(eltype(FourierTransform),ch[5],ch[4],prod(modes)), (3,2,1))

scale = one(eltype(FourierTransform)) / (ch[5] * ch[6])
mapping.layers[4].conv.weight[:,:,:] = permutedims(scale * Flux.glorot_uniform(eltype(FourierTransform),ch[6],ch[5],prod(modes)), (3,2,1))

fno = FourierNeuralOperator(lifting, mapping, project)



test_input = reshape(collect(1:27000).*0.00003, (30,30,10,3,1))

output = fno(test_input)

test_output = reshape(collect(1:27000).*-0.00002, (30,30,10,3,1))


optimizer = Optimisers.Adam(0.001)
state_tree_fno = Flux.setup(optimizer, fno)

loss = 0.0f0
g_fno = Flux.gradient(fno) do p_fno
    result = p_fno(test_input) - test_output
    #result = sum(result, dims=(1,2))
    inner_loss = mean((result).^2)

    ignore() do
        global loss = inner_loss
    end

    inner_loss
end

Flux.update!(state_tree_fno, fno, g_fno[1])

loss


#output_after = fno(test_input)