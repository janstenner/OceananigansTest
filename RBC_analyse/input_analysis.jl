

#first_layer_matrix = agent.policy.approximator.actor.μ.layers[1].weight

# Dense Chain Apprentice
#first_layer_matrix = apprentice.layers[1].weight

# MAT Apprentice
first_layer_matrix = apprentice.encoder.embedding.weight




# try state-to-y
# temp_y = reshape(env.state[:,1], 3,window_size,sensors[2])
# plot(heatmap(z=temp_y[3,:,:]'))
# passes


back_projection = zeros(size(env.state[:,1]))

for i in 1:size(first_layer_matrix)[1]
    for j in 1:size(first_layer_matrix)[2]
        back_projection[j] += abs(first_layer_matrix[i,j])
    end
end



temp_y = reshape(back_projection, 3,window_size,sensors[2]+1)

p = make_subplots(rows=1, cols=3)

add_trace!(p, heatmap(z=temp_y[1,:,:]', coloraxis="coloraxis"), col = 1)
add_trace!(p, heatmap(z=temp_y[2,:,:]', coloraxis="coloraxis"), col = 2)
add_trace!(p, heatmap(z=temp_y[3,:,:]', coloraxis="coloraxis"), col = 3)

colorscale = [[0, "rgb(10, 0, 30)"], [1, "rgb(170, 100, 255)"], ]

layout = Layout(
        plot_bgcolor="#f1f3f7",
        coloraxis = attr(cmin = 0, cmax = maximum(temp_y), colorscale = colorscale),
    )


relayout!(p, layout.fields)

display(p)