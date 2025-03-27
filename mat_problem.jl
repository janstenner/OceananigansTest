using Flux
using Transformers
using Transformers.Layers
using Zygote: ignore
using Statistics
using PlotlyJS
using Optimisers

block_num = 1
dim_model = 22
head_num = 2
head_dim = 11
ffn_dim = 22
drop_out = 0.00


context_size = 64
use_gpu = false
num_epochs = 10
drop_out = 0.1



Base.@kwdef struct SineTransformer
    embedding
    position_encoding
    nl1
    dropout1
    block
    head
end

Flux.@functor SineTransformer


function (st::SineTransformer)(x, obsrep)
    x = st.embedding(x)              # (dm, N, B)
    N = size(x, 2)
    x = x .+ st.position_encoding(1:N) # (dm, N, B)

    x = st.nl1(x)

    x = st.dropout1(x)                # (dm, N, B)

    x = st.block(x, obsrep, Masks.CausalMask(), Masks.CausalMask())     # (dm, N, B)
    x = x[:hidden_state]

    x = st.head(x)                   # (vocab_size, N, B)
    x
end


struct ZeroEncoding
    hidden_size::Int
end

(embed::ZeroEncoding)(x) = zeros(Float32, embed.hidden_size, length(x))


test_transformer = SineTransformer(
    embedding = Dense(1, dim_model, relu, bias = false),
    # position_encoding = Embedding(context_size => dim_model),
    position_encoding = ZeroEncoding(dim_model),
    nl1 = LayerNorm(dim_model, affine = false),
    dropout1 = Dropout(drop_out),
    block = Transformer(RL.CustomTransformerDecoderBlock, block_num, head_num, dim_model, head_dim, ffn_dim; dropout = drop_out),
    head = Dense(dim_model, 1),
)

println("Parameters total: $(sum(length, Flux.params(test_transformer); init=0))")




optimizer = Optimisers.OptimiserChain(Optimisers.ClipNorm(0.1), Optimisers.RMSProp(7e-3))

optimizer_state = Flux.setup(optimizer, test_transformer)




# train

losses = Float32[]
losses_validation = Float32[]

for n in 1:num_epochs
    global losses
  
    global X, Y, obsrep

    X = rand(Float32, 1,12,1)
    X[1,1,1] = 0.0f0
    #obsrep = rand(Float32, dim_model, 12, 1)
    Y = ones(Float32, 1,12,1)
    

    g = gradient(test_transformer) do test_transformer

        global temp_act
        global obsrep
        global μ_expert


        #loss = mean( ((test_transformer(X) - Y)[:, Int(end-(context_size/2)):end ,:]).^2 )
        loss = mean( ((test_transformer(temp_act, obsrep) - μ_expert)).^2 )

        ignore() do
            push!(losses, loss)
        end
    
        loss
    end
    
    Flux.update!(optimizer_state, test_transformer, g[1])


end

display(plot(losses))
#display(plot(losses_validation))



# test!!!!!!

# test_sequence = generate_sequence2()
# context_shown = context_size
# context = reshape(test_sequence[1:context_shown], 1, context_shown, 1)
# if use_gpu
#     context = context |> gpu
# end
# test_transformer_prediction = generate(test_transformer, context; context_size=context_size, max_tokens = 300 - context_shown) |> cpu

# p1 = scatter(y=test_sequence)
# p2 = scatter(y=test_transformer_prediction[:])
# display(plot([p1, p2]))