reset!(env)

colorscale = [[0, "rgb(41, 100, 189)"], [0.19, "rgb(33, 166, 196)"], [0.33, "rgb(246, 230, 170)"], [0.5, "rgb(220, 140, 49)"], [0.62, "rgb(227, 67, 18)"], [1, "rgb(252, 96, 255)"], ]
layout = Layout(
    plot_bgcolor="#FFFFFF",
    coloraxis = attr(cmin = 1, cmax = 2.5, colorscale = colorscale),
)

result = env.y[1,:,:]

Nx = 96
Nz = 64
Lx = 2*pi
Lz = 2

xx = range(0, stop=Lx, length=Nx)
zz = range(0, stop=Lz, length=Nz)

heat = heatmap(x=xx, y=zz, z=result', coloraxis="coloraxis")

plot(heat, layout)


grid = [(x,z) for x in sensor_positions[1] for z in sensor_positions[2]]
X  = getindex.(grid, 1)    # extract all x’s
Y  = getindex.(grid, 2)    # extract all z’s

X = Float32.(X) ./ Nx * Lx
Y = Float32.(Y) ./ Nz * Lz

sensors_scatter = scatter(
  x      = X,
  y      = Y,
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