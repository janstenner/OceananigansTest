using FileIO, JLD2
using PlotlyJS





# temp_only_results is an Array!


global temp_only_results = FileIO.load("RBC_analyse/temp_only_results.jld2","temp_only_results")

FileIO.save("RBC_analyse/temp_only_results.jld2","temp_only_results",temp_only_results)


# plot

p = plot(scatter(y=temp_only_results[1], name="run1", marker_color="rgba(48, 26, 75, 0.8)"))
add_trace!(p, scatter(y=temp_only_results[2], name="run2", marker_color="rgba(109, 177, 191, 0.8)"))
add_trace!(p, scatter(y=temp_only_results[3], name="run3", marker_color="rgba(255, 71, 90, 0.8)"))
add_trace!(p, scatter(y=temp_only_results[4], name="run4", marker_color="rgba(246, 174, 45, 0.8)"))
# index 5 is the successful run
add_trace!(p, scatter(y=temp_only_results[5], name="not temp_only", marker_color="rgba(205, 167, 212, 0.8)"))

relayout!(p, 
            Layout(
                plot_bgcolor = "white",
                font=attr(
                    family="Arial",
                    size=16,
                    color="black"
                ),
                legend=attr(x=0.8, y=0.0,),
                xaxis = attr(gridcolor = "#aaaaaa",
                            linecolor = "#888888"),
                yaxis = attr(gridcolor = "#aaaaaa",
                            linecolor = "#888888"),
                ).fields)
display(p)