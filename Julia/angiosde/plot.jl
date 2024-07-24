import Pkg
    # activate the shared project environment
Pkg.activate(Base.current_project())
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()
using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures, StatsBase, LinearAlgebra,
    Random, Measurements, StatsPlots, HypothesisTests, StatsPlots, Pickle, JLD

# set the default plot size
bm_distance = [x for x in values(JLD.load("jlds/brownian_motion_distance.jld"))][1]
bm_final_pos = [x for x in values(JLD.load("jlds/brownian_motion_final_pos.jld"))][1]

fmb_distance_05 = [x for x in values(JLD.load("jlds/fbm_distance05.jld"))][1]
fmb_final_pos_05 = [x for x in values(JLD.load("jlds/fbm_final_pos05.jld"))][1]

fmb_distance_09 = [x for x in values(JLD.load("jlds/fbm_distance09.jld"))][1]
fmb_final_pos_09 = [x for x in values(JLD.load("jlds/fbm_final_pos09.jld"))][1]

fmb_distance_075 = [x for x in values(JLD.load("jlds/fbm_distance075.jld"))][1]
fmb_final_pos_075 = [x for x in values(JLD.load("jlds/fbm_final_pos075.jld"))][1]

fmb_distance03334 = [x for x in values(JLD.load("jlds/fbm_distance03334.jld"))][1]
fmb_final_pos03334 = [x for x in values(JLD.load("jlds/fbm_final_pos03334.jld"))][1]

# distance

plot1 = violin(['0'], fmb_distance03334,  label = "Fractional Brownian Motion (H = 0.3334)", color = :orange, alpha = 0.5)
violin!(['1'],bm_distance, side = :left, label = "Brownian Motion", color = "blue", alpha = 0.5)
violin!(['1'],fmb_distance_05, side = :right, label = "Fractional Brownian Motion (H = 0.5)", color = "red", alpha = 0.5)
violin!(['2'],fmb_distance_075, label = "Fractional Brownian Motion (H = 0.75)", color = "purple", alpha = 0.5)
violin!(['3'],fmb_distance_09, label = "Fractional Brownian Motion (H = 0.9)", color = "green", alpha = 0.5)
title!("Distribution of the distance \nof the Sprout from the Starting point")
savefig("distance.svg")

plot2 = violin(['0'], fmb_final_pos03334[:,1],  label = "Fractional Brownian Motion (H = 0.3334)", color = :orange, alpha = 0.5)
violin!(['1'],bm_final_pos[:,1], side = :left, label = "Brownian Motion", color = "blue", alpha = 0.5)
violin!(['1'],fmb_final_pos_05[:,1], side = :right, label = "Fractional Brownian Motion (H = 0.5)", color = "red", alpha = 0.5)
violin!(['2'],fmb_final_pos_075[:,1], label = "Fractional Brownian Motion (H = 0.75)", color = "purple", alpha = 0.5)
violin!(['3'],fmb_final_pos_09[:,1], label = "Fractional Brownian Motion (H = 0.9)", color = "green", alpha = 0.5)
title!("Distribution of the final position \nof the Sprout - x-axis")

plot3 = violin(['0'], fmb_final_pos03334[:,2],  label = "Fractional Brownian Motion (H = 0.3334)", color = :orange, alpha = 0.5)
violin!(['1'],bm_final_pos[:,2], side = :left, label = "Brownian Motion", color = "blue", alpha = 0.5)
violin!(['1'],fmb_final_pos_05[:,2], side = :right, label = "Fractional Brownian Motion (H = 0.5)", color = "red", alpha = 0.5)
violin!(['2'],fmb_final_pos_075[:,2], label = "Fractional Brownian Motion (H = 0.75)", color = "purple", alpha = 0.5)
violin!(['3'],fmb_final_pos_09[:,2], label = "Fractional Brownian Motion (H = 0.9)", color = "green", alpha = 0.5)
title!("Distribution of the final position \nof the Sprout - y-axis")

pp = plot(plot2, plot3, layout = (2,1), size = (800, 800))
savefig(pp, "final_pos.svg")