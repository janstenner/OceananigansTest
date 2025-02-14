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


test_input = reshape(collect(1:250).*-0.006, (5,5,5,2,1))

test_output = reshape(collect(250:-1:1).*0.002, (5,5,5,2,1))

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




#------------------------------------------------------------------------------------


#convtest = Conv((1,1), 2 => 2)

convtest = Chain(Conv((1,1), 2 => 2, gelu), Conv((1,1), 2 => 2))

convtest.layers[1].weight[1,1,:,:] = Float32[-1.1714135 -0.17162837; -0.98613584 1.1780858]
convtest.layers[1].bias[:] = Float32[-0.14792423, -0.09325749]
convtest.layers[2].weight[1,1,:,:] = Float32[-1.1714135 -0.17162837; -0.98613584 1.1780858]
convtest.layers[2].bias[:] = Float32[-0.14792423, -0.09325749]

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

convtest.layers[1].weight[1,1,:,:]

convtest.layers[1].bias[:]





#------------------------------------------------------------------------------------




operator_weights = ComplexF32[0.026229387f0 + 0.10053259f0im 0.01752675f0 + 0.08385394f0im; 0.010750301f0 + 0.029626284f0im -0.008826182f0 + 0.042046543f0im;;; -0.032834668f0 + 0.03163823f0im -0.020148955f0 + 0.033979215f0im; -0.02812804f0 + 0.082338676f0im -0.030027697f0 + 0.073727265f0im;;; 0.016512543f0 + 0.09844263f0im 0.008692389f0 + 0.058324374f0im; -0.017100716f0 + 0.07229392f0im -0.05283117f0 + 0.09005813f0im;;; 0.008094089f0 + 0.10484768f0im 0.0214212f0 + 0.08959587f0im; 0.048597567f0 + 0.0037538386f0im 0.020236054f0 + 0.031650145f0im;;; -0.022370525f0 + 0.07120651f0im 0.0035529989f0 + 0.03908349f0im; -0.016054777f0 + 0.038383678f0im -0.0010126173f0 + 0.020552466f0im;;; 0.010483629f0 + 0.056816753f0im -0.047677334f0 + 0.018082688f0im; -0.01593873f0 + 0.008319436f0im 0.03398641f0 + 0.0146105485f0im;;; -0.04407595f0 + 0.07951393f0im 0.014389682f0 + 0.037750956f0im; -0.0415886f0 + 0.021245018f0im 0.009691542f0 + 0.053529415f0im;;; -0.03459335f0 + 0.097394556f0im -0.012144056f0 + 0.07765343f0im; -0.042812895f0 + 0.017810017f0im 0.004505572f0 + 0.049285274f0im;;; -0.023788774f0 + 0.06661265f0im 0.013829015f0 + 0.011470425f0im; 0.026985377f0 + 0.0042261477f0im -0.051237177f0 + 0.013503404f0im;;; -0.029286517f0 + 0.062471945f0im -0.033324506f0 + 0.010113707f0im; -0.014423192f0 + 0.09087f0im 0.0051092664f0 + 0.04867593f0im;;; -0.021925729f0 + 0.06564635f0im 0.002008759f0 + 0.071877025f0im; -0.05197615f0 + 0.09997875f0im -0.030926986f0 + 0.03355207f0im;;; 0.024988774f0 + 0.096475564f0im 0.007125841f0 + 0.009331589f0im; -0.025977971f0 + 0.07231698f0im -0.01829255f0 + 0.02197107f0im;;; -0.012770065f0 + 0.096389465f0im 0.050012525f0 + 0.043570094f0im; 0.014296946f0 + 0.02379103f0im -0.0057297414f0 + 0.0816557f0im;;; -0.016716668f0 + 0.07282261f0im 0.009229406f0 + 0.082883485f0im; 0.03508331f0 + 0.011818001f0im -0.010376185f0 + 0.0114437705f0im;;; -0.006957895f0 + 0.036807455f0im 0.013071049f0 + 0.019494742f0im; -0.051629886f0 + 0.013817407f0im -0.041066043f0 + 0.035899613f0im;;; -0.00745074f0 + 0.0258138f0im -0.053002197f0 + 0.043034513f0im; -0.026380926f0 + 0.04754025f0im -0.044581372f0 + 0.0077546774f0im;;; -0.050270595f0 + 0.06544204f0im -0.029836718f0 + 0.07997588f0im; 0.018672239f0 + 0.0384231f0im 0.016916573f0 + 0.045047503f0im;;; 0.0073864385f0 + 0.061639015f0im 0.0018953423f0 + 0.104636654f0im; 0.042868253f0 + 0.09224281f0im 0.0012210244f0 + 0.0569083f0im;;; -0.051999535f0 + 0.013340777f0im 0.028888935f0 + 0.05769805f0im; -0.000297857f0 + 0.049601417f0im 0.036126897f0 + 0.06435076f0im;;; -0.052652057f0 + 0.08083344f0im -0.007889265f0 + 0.010136428f0im; -0.016933741f0 + 0.10076447f0im -0.026790967f0 + 0.06724669f0im;;; -0.036487557f0 + 0.07865125f0im -0.0027419392f0 + 0.08720084f0im; -0.048469994f0 + 0.066837154f0im 0.0028131153f0 + 0.0671118f0im;;; -0.0156145245f0 + 0.024119334f0im -0.0474221f0 + 0.06448642f0im; 0.03866333f0 + 0.06316359f0im -0.03936478f0 + 0.0034763203f0im;;; -0.0005027886f0 + 0.077525295f0im 0.03420236f0 + 0.020255765f0im; 0.051978916f0 + 0.008037425f0im -0.035497565f0 + 0.025905924f0im;;; 0.017604018f0 + 0.09003501f0im 0.03350739f0 + 0.06374622f0im; -0.027161976f0 + 0.024537452f0im -0.029282138f0 + 0.10160388f0im;;; -0.020150011f0 + 0.031451233f0im 0.023309767f0 + 0.07305199f0im; -0.03253026f0 + 0.04380403f0im 0.041633997f0 + 0.050426822f0im;;; -0.028883528f0 + 0.011864187f0im -0.0033403027f0 + 0.04647034f0im; 0.04825431f0 + 0.020939454f0im 0.039920416f0 + 0.06497868f0im;;; 0.012564619f0 + 0.031101877f0im 0.016741613f0 + 0.081643105f0im; 0.048989095f0 + 0.046513f0im 0.05179778f0 + 0.09353385f0im;;; 0.0076079858f0 + 0.061493497f0im 0.022849385f0 + 0.09847001f0im; 0.043135278f0 + 0.021092704f0im 0.029410608f0 + 0.07833485f0im;;; -0.010557112f0 + 0.08831124f0im 0.041165058f0 + 0.024545647f0im; 0.017422762f0 + 0.08902272f0im -0.003014387f0 + 0.09415155f0im;;; -0.023271473f0 + 0.06951431f0im 0.03850451f0 + 0.07192946f0im; -0.004420926f0 + 0.099626765f0im -0.016906973f0 + 0.0072816825f0im;;; -0.052788567f0 + 0.096559696f0im 0.0134868715f0 + 0.069595225f0im; 0.047538638f0 + 0.093824275f0im 0.023811242f0 + 0.09444822f0im;;; 0.01607905f0 + 0.07855638f0im -0.0071079037f0 + 0.044281278f0im; -0.0074734106f0 + 0.06517735f0im -0.048041902f0 + 0.06274803f0im;;; -0.036332205f0 + 0.10616623f0im -0.039882895f0 + 0.078982145f0im; -0.037666816f0 + 0.1056348f0im 0.008529552f0 + 0.10177374f0im;;; 0.014088997f0 + 0.079565234f0im -0.016036104f0 + 0.00079489534f0im; 0.009630615f0 + 0.013482519f0im 0.040712122f0 + 0.09169951f0im;;; 0.03513188f0 + 0.04291985f0im -0.01797222f0 + 0.06945307f0im; 0.013298079f0 + 0.101801254f0im 0.023826892f0 + 0.040915478f0im;;; 0.011449685f0 + 0.0595539f0im 0.052235804f0 + 0.017278707f0im; 0.043335337f0 + 0.04947853f0im -0.03613236f0 + 0.021796122f0im;;; 0.015197406f0 + 0.08756588f0im 0.039608095f0 + 0.01621327f0im; -0.017356122f0 + 0.058403246f0im 0.006935015f0 + 0.094449975f0im;;; -0.050937954f0 + 0.08081784f0im 0.026594855f0 + 0.083077066f0im; 0.026391596f0 + 0.07688066f0im -0.022846868f0 + 0.007979332f0im;;; -0.016495407f0 + 0.087691285f0im 0.03065694f0 + 0.03151391f0im; 0.023710843f0 + 0.102594614f0im -0.02668543f0 + 0.10618205f0im;;; -0.04415476f0 + 0.034520674f0im -0.003508058f0 + 0.07348726f0im; 0.01744901f0 + 0.069103934f0im -0.013758093f0 + 0.041436445f0im;;; 0.018062647f0 + 0.092591986f0im 0.051837802f0 + 0.04348008f0im; -0.048886027f0 + 0.08268878f0im -0.0119235385f0 + 0.05167908f0im;;; -0.0062079025f0 + 0.0916168f0im 0.033988692f0 + 0.020964235f0im; 0.0018555035f0 + 0.019303344f0im 0.047856838f0 + 0.010724607f0im;;; 0.004208865f0 + 0.010824445f0im 0.044116866f0 + 0.011715328f0im; -0.01825294f0 + 0.06734915f0im -0.0138441f0 + 0.09725602f0im;;; 0.028639698f0 + 0.07605762f0im -0.009552837f0 + 0.048559833f0im; -0.046468988f0 + 0.08494699f0im 0.03289292f0 + 0.08114147f0im;;; -0.0054872143f0 + 0.060113233f0im 0.02769395f0 + 0.031818796f0im; -0.04741858f0 + 0.057529677f0im -0.028053477f0 + 0.030697249f0im;;; -0.0050668227f0 + 0.09725547f0im 0.048163198f0 + 0.060555633f0im; 0.022510456f0 + 0.069112524f0im 0.005334016f0 + 0.0621316f0im;;; 0.02757517f0 + 0.08215617f0im -0.021580046f0 + 0.001799024f0im; 0.04359846f0 + 0.08455515f0im 0.015891338f0 + 0.07556874f0im;;; -0.04064869f0 + 0.038120102f0im 0.034221336f0 + 0.038125806f0im; 0.029927718f0 + 0.070941806f0im -0.0088536125f0 + 0.047941282f0im;;; -0.035624713f0 + 0.0309236f0im 0.011492942f0 + 0.062979415f0im; 0.022922257f0 + 0.016024133f0im 0.049964815f0 + 0.09633509f0im;;; 0.01566157f0 + 0.10003619f0im -0.031910256f0 + 0.07657183f0im; 0.017190864f0 + 0.04148011f0im 0.040455636f0 + 0.026465332f0im;;; 0.0499934f0 + 0.030842327f0im 0.043804616f0 + 0.036374655f0im; 0.04191144f0 + 0.079731025f0im 0.050261494f0 + 0.047496643f0im;;; -0.04489778f0 + 0.09101046f0im 0.045132723f0 + 0.042097222f0im; 0.024144128f0 + 0.056086194f0im 0.007942657f0 + 0.013267161f0im;;; -0.004066424f0 + 0.07382404f0im -0.039750393f0 + 0.033053998f0im; 0.037807047f0 + 0.0047222013f0im -0.03747097f0 + 0.023594217f0im;;; -0.039648958f0 + 0.10636737f0im -0.051060997f0 + 0.05109473f0im; -0.009440101f0 + 0.07626383f0im 0.009409208f0 + 0.106319144f0im;;; -0.010492658f0 + 0.077252194f0im -0.048537947f0 + 0.042311177f0im; 0.013870093f0 + 0.0808403f0im 0.03708656f0 + 0.04962556f0im;;; 0.026264021f0 + 0.015626559f0im -0.012994554f0 + 0.079017f0im; 0.027071955f0 + 0.045858886f0im 0.010301305f0 + 0.008605799f0im;;; 0.037234224f0 + 0.038799025f0im 0.005986006f0 + 0.08088429f0im; -0.011982668f0 + 0.06041623f0im 0.0096442755f0 + 0.06511804f0im;;; 0.05317412f0 + 0.08571054f0im -0.032688566f0 + 0.0058659306f0im; -0.03232681f0 + 0.06290201f0im -0.014299431f0 + 0.082518764f0im;;; 0.028635943f0 + 0.10202065f0im 0.04896524f0 + 0.055549186f0im; -0.020603156f0 + 0.06743177f0im 0.051818073f0 + 0.07837556f0im;;; -0.048367646f0 + 0.09204973f0im -0.048933752f0 + 0.023994861f0im; -0.021773744f0 + 0.042927414f0im 0.008526953f0 + 0.05901405f0im;;; -0.042317502f0 + 0.065370426f0im 0.016122516f0 + 0.029491467f0im; -0.01762625f0 + 0.08020022f0im 0.050069556f0 + 0.06719021f0im;;; -0.020966725f0 + 0.004566277f0im -0.037329547f0 + 0.03522539f0im; 0.010801863f0 + 0.040117234f0im 0.048170395f0 + 0.09311588f0im;;; 0.030754631f0 + 0.08385001f0im 0.04138285f0 + 0.04454746f0im; -0.02399675f0 + 0.07468416f0im -0.039099373f0 + 0.06378463f0im;;; 0.05034113f0 + 0.10386062f0im -0.04009612f0 + 0.047119394f0im; -0.024176303f0 + 0.08703861f0im 0.05195851f0 + 0.034091726f0im]



ch2=(1,2,2,2,2,2, 4, 1)
modes2=(4,4,4,)
σ = gelu

lifting2 = Conv((1,1,1), ch2[1]=>ch2[2])
mapping2 = Chain(OperatorKernel(ch2[2] => ch2[3], modes2, Transform, σ; permuted = true),
                OperatorKernel(ch2[3] => ch2[4], modes2, Transform, σ; permuted = true),
                OperatorKernel(ch2[4] => ch2[5], modes2, Transform, σ; permuted = true),
                OperatorKernel(ch2[5] => ch2[6], modes2, Transform, σ; permuted = true))
project2 = Chain(Conv((1,1,1), ch2[6]=>ch2[7], σ),
                Conv((1,1,1), ch2[7]=>ch2[8]))


fno2 = FourierNeuralOperator(lifting2, mapping2, project2)




fno2.lifting_net.weight[:,:,:,:,:] = Float32[-0.38120607;;;;; -0.16277888]
fno2.lifting_net.bias[:] = Float32[-0.9021771, 0.41493654]

fno2.integral_kernel_net.layers[1].linear.weight[:,:,:,:,:] = Float32[-1.0933883;;;; -0.49589157;;;;; 0.14858606;;;; 0.28509623]
fno2.integral_kernel_net.layers[1].linear.bias[:] = Float32[0.25514624, 0.4911342]
fno2.integral_kernel_net.layers[1].conv.weight[:,:,:] = permutedims(operator_weights, (3,2,1))

fno2.integral_kernel_net.layers[2].linear.weight[:,:,:,:,:] = Float32[-1.0933883;;;; -0.49589157;;;;; 0.14858606;;;; 0.28509623]
fno2.integral_kernel_net.layers[2].linear.bias[:] = Float32[0.25514624, 0.4911342]
fno2.integral_kernel_net.layers[2].conv.weight[:,:,:] = permutedims(operator_weights, (3,2,1))

fno2.integral_kernel_net.layers[3].linear.weight[:,:,:,:,:] = Float32[-1.0933883;;;; -0.49589157;;;;; 0.14858606;;;; 0.28509623]
fno2.integral_kernel_net.layers[3].linear.bias[:] = Float32[0.25514624, 0.4911342]
fno2.integral_kernel_net.layers[3].conv.weight[:,:,:] = permutedims(operator_weights, (3,2,1))

fno2.integral_kernel_net.layers[4].linear.weight[:,:,:,:,:] = Float32[-1.0933883;;;; -0.49589157;;;;; 0.14858606;;;; 0.28509623]
fno2.integral_kernel_net.layers[4].linear.bias[:] = Float32[0.25514624, 0.4911342]
fno2.integral_kernel_net.layers[4].conv.weight[:,:,:] = permutedims(operator_weights, (3,2,1))

fno2.project_net.layers[1].weight[:,:,:,:,:] = Float32[-0.42080823;;;; -0.18447632;;;;; -0.68774295;;;; 1.0447022;;;;; -0.59104174;;;; -0.62169915;;;;; 0.054271773;;;; -0.7646005]
fno2.project_net.layers[1].bias[:] = Float32[-0.07961995, -0.39764675, -0.12689102, -0.29209343]
fno2.project_net.layers[2].weight[:,:,:,:,:] = Float32[0.47013336;;;; 0.0027586299;;;; 0.37975815;;;; 0.14138512;;;;;]
fno2.project_net.layers[2].bias[:] = Float32[0.09048933]


test_input = reshape(collect(1:150).*0.003, (6,5,5,1,1))

test_output = reshape(collect(1:150).*-0.002, (6,5,5,1,1))

output_before = fno2(test_input)

temp = fno2.lifting_net(test_input)

temp2 = fno2.integral_kernel_net.layers[1](temp)
temp3 = fno2.integral_kernel_net.layers[2](temp2)
temp4 = fno2.integral_kernel_net.layers[3](temp3)
temp5 = fno2.integral_kernel_net.layers[4](temp4)

temp6 = fno2.project_net(temp5)


optimizer = Optimisers.Adam(0.001)
state_tree_fno2 = Flux.setup(optimizer, fno2)

loss = 0.0f0
g_fno2 = Flux.gradient(fno2) do p_fno2
    result = p_fno2(test_input) - test_output
    #result = sum(result, dims=(1,2))
    inner_loss = mean((result).^2)

    ignore() do
        global loss = inner_loss
    end

    inner_loss
end

Flux.update!(state_tree_fno2, fno2, g_fno2[1])

loss


output_after = fno2(test_input)



#-------------------------------------------------------------------------------------------------------------------------------



function compare_parameters(x,y=nothing)
    println("----------------")
    if isnothing(y)

        println("$(mean(x))")
        println("$(std(x))")
        if(typeof(x[1])==ComplexF32)
            println("$(minimum(real(x)))")
            println("$(maximum(real(x)))")
            println("$(minimum(imag(x)))")
            println("$(maximum(imag(x)))")
        else
            println("$(minimum(x))")
            println("$(maximum(x))")
        end

    else

        println("$(mean(x))   vs   $(mean(y))")
        println("$(std(x))   vs   $(std(y))")
        if(typeof(x[1])==ComplexF32)
            println("$(minimum(real(x)))   vs   $(minimum(real(y)))")
            println("$(maximum(real(x)))   vs   $(maximum(real(y)))")
            println("$(minimum(imag(x)))   vs   $(minimum(imag(y)))")
            println("$(maximum(imag(x)))   vs   $(maximum(imag(y)))")
        else
            println("$(minimum(x))   vs   $(minimum(y))")
            println("$(maximum(x))   vs   $(maximum(y))")
        end
        
    end
    println("----------------")
end

compare_parameters(lifting.weight, ps.layer_1.weight)    # min and max dont match
compare_parameters(lifting.bias, ps.layer_1.bias)

compare_parameters(mapping.layers[1].linear.weight, ps.layer_2.layer_1.layer_1.weight)
compare_parameters(mapping.layers[1].linear.bias, ps.layer_2.layer_1.layer_1.bias)
compare_parameters(mapping.layers[1].conv.weight, permutedims(ps.layer_2.layer_1.layer_2.weight[:,:,:], (3,2,1))) # min and max of imag shifted

compare_parameters(mapping.layers[1].conv.weight)

# compare_parameters(mapping.layers[2].linear.weight, ps.layer_2.layer_2.layer_1.weight)
# compare_parameters(mapping.layers[2].linear.bias, ps.layer_2.layer_2.layer_1.bias)
# compare_parameters(mapping.layers[2].conv.weight, permutedims(ps.layer_2.layer_2.layer_2.weight[:,:,:], (3,2,1)))

# compare_parameters(mapping.layers[3].linear.weight, ps.layer_2.layer_3.layer_1.weight)
# compare_parameters(mapping.layers[3].linear.bias, ps.layer_2.layer_3.layer_1.bias)
# compare_parameters(mapping.layers[3].conv.weight, permutedims(ps.layer_2.layer_3.layer_2.weight[:,:,:], (3,2,1)))

# compare_parameters(mapping.layers[4].linear.weight, ps.layer_2.layer_4.layer_1.weight)
# compare_parameters(mapping.layers[4].linear.bias, ps.layer_2.layer_4.layer_1.bias)
# compare_parameters(mapping.layers[4].conv.weight, permutedims(ps.layer_2.layer_4.layer_2.weight[:,:,:], (3,2,1)))

compare_parameters(project.layers[1].weight, ps.layer_3.layer_1.weight)    # min and max dont match
compare_parameters(project.layers[1].bias, ps.layer_3.layer_1.bias)

compare_parameters(project.layers[2].weight, ps.layer_3.layer_2.weight)    # min and max dont match
compare_parameters(project.layers[2].bias, ps.layer_3.layer_2.bias)