

first_layer_matrix = agent.policy.approximator.actor.μ.layers[1].weight




# try state-to-y
temp_y = reshape(env.state[:,1], 3,window_size,sensors[2])
plot(heatmap(z=temp_y[3,:,:]'))
# passes


back_projection = zeros(size(env.state[:,1]))

for i in 1:size(first_layer_matrix)[1]
    for j in 1:size(first_layer_matrix)[2]
        back_projection[j] += abs(first_layer_matrix[i,j])
    end
end



temp_y = reshape(back_projection, 3,window_size,sensors[2])

p = make_subplots(rows=1, cols=3)

add_trace!(p, heatmap(z=temp_y[1,:,:]'), col = 1)
add_trace!(p, heatmap(z=temp_y[2,:,:]'), col = 2)
add_trace!(p, heatmap(z=temp_y[3,:,:]'), col = 3)

display(p)