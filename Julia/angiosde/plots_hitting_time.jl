import Pkg
    # activate the shared project environment
Pkg.activate(Base.current_project())
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()
using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures, StatsBase, LinearAlgebra,
    Random, Measurements, StatsPlots, HypothesisTests, StatsPlots, Pickle, JLD

current_dir = pwd()
# hit_time_variables
ht_var = current_dir * "/jlds/hiting_times/"
ht_var_files = readdir(ht_var, join = true)

read_keys = []
read_ht = []
for file in ht_var_files
    
    dict_var = JLD.load(file)
   
    push!(read_keys, collect(keys(dict_var))[1])
    push!(read_ht, collect(values(dict_var))[1])
end 

for i in 1:length(read_keys)
    println(size(read_ht[i]))
end

Plots.plot(title = "Hitting Time Distribution") 
for i in 1:length(read_keys)
    violin!([read_keys[i]], read_ht[i], alpha = 0.5)
end
