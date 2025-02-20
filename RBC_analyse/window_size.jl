using FileIO, JLD2





# window_size_results is a Dict!


global window_size_results = FileIO.load("RBC_analyse/window_size_results.jld2","window_size_results")

FileIO.save("RBC_analyse/window_size_results.jld2","window_size_results",window_size_results)