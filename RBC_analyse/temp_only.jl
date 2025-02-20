using FileIO, JLD2





# temp_only_results is an Array!


global temp_only_results = FileIO.load("RBC_analyse/temp_only_results.jld2","temp_only_results")

FileIO.save("RBC_analyse/temp_only_results.jld2","temp_only_results",temp_only_results)