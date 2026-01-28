using FileIO, JLD2
using PlotlyJS





# window_size_results is a Dict!


global window_size_results = FileIO.load("RBC_analyse/window_size_results.jld2","window_size_results")

# FileIO.save("RBC_analyse/window_size_results.jld2","window_size_results",window_size_results)




# plot

function plot_it()
    global p
    relayout!(p, 
                Layout(
                    plot_bgcolor = "white",
                    font=attr(
                        family="Arial",
                        size=16,
                        color="black"
                    ),
                    legend=attr(x=1.0, y=0.0,),
                    xaxis = attr(gridcolor = "#aaaaaa",
                                linecolor = "#888888"),
                    yaxis = attr(gridcolor = "#aaaaaa",
                                linecolor = "#888888"),
                    ).fields)
    display(p)
end


p = plot()
for key in 47:-2:39

    add_trace!(p, scatter(y=window_size_results[key], name="$(key)"))

end
plot_it()


p = plot()
for key in 37:-2:29

    add_trace!(p, scatter(y=window_size_results[key], name="$(key)"))

end
plot_it()


p = plot()
for key in 27:-2:19

    add_trace!(p, scatter(y=window_size_results[key], name="$(key)"))

end
plot_it()


p = plot()
for key in 17:-2:9

    add_trace!(p, scatter(y=window_size_results[key], name="$(key)"))

end
plot_it()


p = plot()
for key in 7:-2:3

    add_trace!(p, scatter(y=window_size_results[key], name="$(key)"))

end
plot_it()


p = plot()

add_trace!(p, scatter(y=window_size_results[701], name="run 1"))
add_trace!(p, scatter(y=window_size_results[702], name="run 2"))
add_trace!(p, scatter(y=window_size_results[703], name="run 3"))

plot_it()