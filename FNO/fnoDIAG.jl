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

test_input = Float32[-0.5 -0.7; -0.2 1.1;;; 0.5 0.7; -1.4 0.1;;;;]

convtest(test_input)

test_output = Float32[1.0 -1.0; -0.0 1.0;;; 2.5 2.0; -1.0 0.0;;;;]


optimizer = Optimisers.Adam(0.01)
state_tree_convtest = Flux.setup(optimizer, convtest)

loss = 0.0f0
g_convtest = Flux.gradient(convtest) do ct
    result = ct(test_input) - test_output
    #result = sum(result, dims=(1,2))
    inner_loss = mean((result).^2)

    ignore() do
        global loss = inner_loss
    end

    inner_loss
end

Flux.update!(state_tree_convtest, convtest, g_convtest[1])

loss

convtest.weight[1,1,:,:]

convtest.bias[:]


# bias computation for conv layers in Flux:

function custom_uniform(rng::AbstractRNG, fan_in, dims...)
    bound = Float32(1/ sqrt(fan_in)) # fan_in
    return (rand(rng, Float32, dims...) .- 0.5f0) .* 2bound
end

custom_uniform(fan_in, dims...; kwargs...) = custom_uniform(Random.GLOBAL_RNG, fan_in, dims...; kwargs...)


fan_in = first(Flux.nfan(size(convtest.weight)))
custom_uniform(fan_in, size(convtest.bias)...)

convtest2 = Conv((1,1,1), ch2[6]=>ch2[7], σ)
fan_in = first(Flux.nfan(size(convtest2.weight)))
custom_uniform(fan_in, size(convtest2.bias)...)

convtest3 = Conv((1,1,1), ch2[7]=>ch2[8])
fan_in = first(Flux.nfan(size(convtest3.weight)))
custom_uniform(fan_in, size(convtest3.bias)...)



#--------------------------------------------------------------------------------------


ok = OperatorKernel(2 => 2, (3,3,3), Transform, σ; permuted = true)
okconv = ok.conv
println(okconv.weight)

weights_from_flux = ComplexF32[-0.030560905f0 + 0.04372935f0im 0.01915261f0 - 0.030111514f0im; 0.042418927f0 + 0.01092832f0im 0.018298335f0 - 0.009250117f0im; 0.03642035f0 + 0.0431377f0im -0.044067234f0 + 0.024916008f0im; -0.015749786f0 + 0.048252676f0im 0.021642217f0 - 0.04505753f0im; 0.008768308f0 - 0.008788868f0im 0.0008172766f0 - 0.024791492f0im; 0.030345753f0 + 0.005071509f0im 0.008754575f0 - 0.04880611f0im; -0.041313015f0 - 0.031145707f0im -0.03287885f0 - 0.012858504f0im; -0.037475023f0 - 0.03405146f0im 0.050672617f0 + 0.027790241f0im; 0.049064692f0 - 0.036998287f0im 0.024299856f0 + 0.019165367f0im; -0.00190803f0 + 0.010604744f0im -0.04970548f0 - 0.010421855f0im; -0.0050842655f0 - 0.032341238f0im 0.014498621f0 - 0.018965555f0im; 0.029277034f0 + 0.0074868468f0im -0.04482101f0 + 0.008909015f0im; -0.002480005f0 + 0.00887456f0im -0.030337976f0 + 0.033779386f0im; 0.0062758345f0 - 0.03201661f0im 0.022720475f0 + 0.013435748f0im; 0.04958437f0 + 0.020257166f0im 0.022709208f0 + 0.010161907f0im; 0.01701776f0 + 0.027048415f0im -0.006360121f0 + 0.02254269f0im; -0.04229607f0 - 0.035868492f0im -0.050664607f0 - 0.021614436f0im; -0.040716294f0 - 0.03688698f0im -0.057312198f0 + 0.007873888f0im; 0.0004920434f0 + 0.020522783f0im -0.015368146f0 + 0.014249603f0im; 0.037897233f0 - 0.02346511f0im 0.0046768393f0 + 0.03631083f0im; -0.028136687f0 - 0.039303504f0im -0.040131893f0 + 0.011270938f0im; 0.0509575f0 + 0.057353783f0im 0.049477905f0 - 0.0077606677f0im; 0.05791499f0 + 0.0065005263f0im 0.041787356f0 - 0.03843315f0im; 0.042492114f0 + 0.031625934f0im -0.037765726f0 - 0.008004228f0im; -0.023778824f0 + 0.052113287f0im 0.003178968f0 + 0.031100307f0im; 0.020586425f0 - 0.013723014f0im 0.040758952f0 - 0.0045130565f0im; -0.015195048f0 - 0.015609669f0im -0.021882966f0 - 0.011877487f0im;;; 0.025898458f0 - 0.02403471f0im -0.025871668f0 + 0.03843465f0im; -0.01843409f0 + 0.026128074f0im 0.017718185f0 + 0.04723252f0im; -0.039400987f0 - 0.003875494f0im -0.029231628f0 - 0.055939537f0im; 0.019568775f0 - 0.003142722f0im 0.036331546f0 + 0.009791598f0im; -0.03517353f0 - 0.029303791f0im 0.030199565f0 - 0.040092543f0im; 0.011762876f0 - 0.0419515f0im 0.03368895f0 - 0.010152796f0im; -0.031346094f0 - 0.015873866f0im -0.05598319f0 - 0.00863714f0im; -0.023492899f0 - 0.0012797961f0im -0.012313153f0 + 0.021792373f0im; 0.04561936f0 + 0.013623885f0im 0.05283206f0 - 0.040239107f0im; -0.018156271f0 - 0.015376084f0im -0.010964861f0 - 0.009984427f0im; 0.02198856f0 - 0.010771203f0im 0.008366184f0 + 0.011795512f0im; 0.011301192f0 - 0.046596207f0im 0.032853287f0 - 0.033272702f0im; 0.012108776f0 - 0.021573251f0im 0.043423764f0 + 0.002128177f0im; -0.049595524f0 + 0.053188115f0im 0.04167739f0 - 0.03647932f0im; -0.01604307f0 + 0.028788919f0im -0.0140036f0 - 0.043671027f0im; 0.04099775f0 + 0.04663383f0im -0.017305525f0 - 0.048596505f0im; -0.004673074f0 + 0.04641353f0im 0.0063211f0 + 0.02731429f0im; -0.057189018f0 + 0.025969202f0im -0.044659153f0 + 0.029224442f0im; 0.021763088f0 - 0.029280217f0im 0.047085095f0 - 0.015512963f0im; -0.006014784f0 + 0.03744776f0im -0.04511261f0 - 0.040396716f0im; 0.05865364f0 + 0.021752466f0im -0.022324428f0 - 0.017131345f0im; 0.029833583f0 + 0.00069290824f0im 0.035650518f0 + 0.02925549f0im; 0.0075350488f0 + 0.04499061f0im 0.04826563f0 + 0.048340462f0im; -0.04937079f0 + 0.052026276f0im 0.027827302f0 + 0.017799802f0im; -0.01289938f0 - 0.010421146f0im 0.044920463f0 + 0.052969147f0im; 0.014340774f0 + 0.011515074f0im -0.03988697f0 - 0.021322701f0im; 0.058163803f0 - 0.007738f0im -0.03824999f0 - 0.035214886f0im]

okconv.weight[:,:,:] = weights_from_flux

test_input = reshape(collect(1:250).*0.003, (5,5,5,2,1))

NeuralOperators.operator_conv(okconv, test_input)

test_output = reshape(collect(1:250).*-0.002, (5,5,5,2,1))



optimizer = Optimisers.Adam(0.01)
state_tree_okconv = Flux.setup(optimizer, okconv)

loss = 0.0f0
g_okconv = Flux.gradient(okconv) do okc
    result = NeuralOperators.operator_conv(okc, test_input) - test_output
    #result = sum(result, dims=(1,2))
    inner_loss = mean((result).^2)

    ignore() do
        global loss = inner_loss
    end

    inner_loss
end

Flux.update!(state_tree_okconv, okconv, g_okconv[1])

loss

flux_change = okconv.weight - weights_from_flux

lux_change = ComplexF32[-0.009999996f0 + 0.0f0im -0.009999999f0 + 0.0f0im; -0.009989429f0 - 2.304744f-5im -0.009989429f0 - 2.304744f-5im; -0.009917017f0 - 0.0010240823f0im -0.009917017f0 - 0.0010240804f0im; -0.009999055f0 - 2.3856759f-5im -0.009999055f0 - 2.3856759f-5im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; -0.009991348f0 + 0.00034917518f0im -0.009991348f0 + 0.00034917518f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; -0.009999962f0 - 1.4267862f-6im -0.009999961f0 - 1.4267862f-6im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; -0.009996426f0 - 0.00026364066f0im -0.009996426f0 - 0.0002636416f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im;;; -0.01f0 + 0.0f0im -0.009999998f0 + 0.0f0im; -0.009928493f0 - 0.0010937154f0im -0.009928494f0 - 0.0010937154f0im; -0.009917185f0 + 0.0009919114f0im -0.009917187f0 + 0.0009919107f0im; -0.009999001f0 - 4.7855312f-5im -0.009999001f0 - 4.785508f-5im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; -0.009995487f0 + 0.00019662827f0im -0.009995487f0 + 0.00019662827f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; -0.009998072f0 + 0.00019441359f0im -0.009998072f0 + 0.00019441359f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; -0.0099948095f0 + 0.00031928718f0im -0.009994809f0 + 0.0003192881f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im; 0.0f0 + 0.0f0im 0.0f0 + 0.0f0im]

minimum(real(flux_change - lux_change))
minimum(imag(flux_change - lux_change))