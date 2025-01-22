GC.gc(true)
CUDA.reclaim()




ch2=(3,3,3,3,3,3, 128, 3)
modes2=(16,16,fno_input_timesteps,)
σ = gelu

lifting2 = Conv((1,1,1), ch2[1]=>ch2[2])
mapping2 = Chain(OperatorKernel(ch2[2] => ch2[3], modes2, Transform, σ; permuted = true),
                OperatorKernel(ch2[3] => ch2[4], modes2, Transform, σ; permuted = true),
                OperatorKernel(ch2[4] => ch2[5], modes2, Transform, σ; permuted = true),
                OperatorKernel(ch2[5] => ch2[6], modes2, Transform; permuted = true))
project2 = Chain(Conv((1,1,1), ch2[6]=>ch2[7], σ),
                Conv((1,1,1), ch2[7]=>ch2[8]))


fno2 = FourierNeuralOperator(lifting2, mapping2, project2)


convv = fno2.integral_kernel_net.layers[1].conv
tform = convv.transform

test_input = permutedims(results[:,:,:,1:1+fno_input_timesteps-1],(1,2,4,3))
test_input = reshape(test_input, (Nx,Nz,10,3,1))



x_t = NeuralOperators.transform(tform, test_input)
x_tr = NeuralOperators.truncate_modes(tform, x_t)


plot(heatmap(z=real(x_tr)[:,:,8,1,1]))


x_p = NeuralOperators.apply_pattern(x_tr, convv.weight)

x_tr2 = Array(x_tr)
x_padded = NeuralOperators.pad_modes(x_tr2, (size(x_t)[1:(end - 2)]..., size(x_tr2)[(end - 1):end]...))

plot(heatmap(z=real(x_padded)[:,:,8,1,1]))

x_inv = NeuralOperators.inverse(tform, x_padded, size(test_input))

plot(heatmap(z=test_input[:,:,8,1,1]'))
plot(heatmap(z=x_inv[:,:,8,1,1]'))


#------------------------------------------------------------------------------------


convtest = Conv((1,1), 2 => 2)

convtest.weight[1,1,:,:] = Float32[-1.1714135 -0.17162837; -0.98613584 1.1780858]
convtest.bias[:] = Float32[-0.14792423, -0.09325749]

test_input = Float32[-0.5 -0.7; -0.2 1.1 ;;; 0.5 0.7; -1.4 0.1 ]