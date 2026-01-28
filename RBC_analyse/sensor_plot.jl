reset!(env)

colorscale = [[0, "rgb(34, 74, 168)"], [0.25, "rgb(224, 224, 180)"], [0.5, "rgb(156, 33, 11)"], [1, "rgb(226, 63, 161)"], ]
layout = Layout(
    plot_bgcolor="#f1f3f7",
    coloraxis = attr(cmin = 1, cmid = 2.5, cmax = 3, colorscale = colorscale),
)

result = env.y[1,:,:]

heat = heatmap(z=result', coloraxis="coloraxis")


grid = [(x,z) for x in sensor_positions[1] for z in sensor_positions[2]]
X  = getindex.(grid, 1)   # extract all x’s
Y  = getindex.(grid, 2)   # extract all z’s

sensors_scatter = scatter(
  x      = X .-1,
  y      = Y .-1,
  mode   = "markers",
  marker = attr(
    symbol   = "circle",
    size     = 4,
    color    = "black",
    opacity  = 0.5
  ),
  showlegend = false
)


p = plot([heat, sensors_scatter], layout)
display(p)