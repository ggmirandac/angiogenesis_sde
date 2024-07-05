import Pkg
    # activate the shared project environment
Pkg.activate(Base.current_project())
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()
using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures, StatsBase, LinearAlgebra,
    Random, Measurements, HypothesisTests, StatsPlots

"""
Given the model of Stokes and Lauffenburger, the model uses a different 
white noise source for each dimension. So, let's compare what we would get only from 
the random source
"""


function cholesky_fbm(n_steps::Int, n_dim::Int, H::Float64, T::Float64)
    dt = T / n_steps
    t = range(0, T, length=n_steps+1)
    
    # Create covariance matrix
    cov_matrix = [0.5 * (abs(ti)^(2*H) + abs(tj)^(2*H) - abs(ti-tj)^(2*H)) for ti in t, tj in t]
    
    # Add a small positive value to the diagonal for regularization
    epsilon = 1e-8 * maximum(diag(cov_matrix))
    cov_matrix += epsilon * I
    
    # Cholesky decomposition
    L = cholesky(Symmetric(cov_matrix)).L
    
    # Generate standard normal random variables
    Z = randn(n_steps+1, n_dim) 

    # Generate fractional Brownian motion
    X = L * Z
    X *= dt ^ H
    
    return X
end

function brownian_motion(n_steps::Int, n_dim::Int, T::Float64)
    dt = T / n_steps
    dW = randn(n_steps, n_dim) * dt
    W = cumsum(dW, dims=1)
    return vcat(zeros(n_dim)', W)
end

# Example usage
n_steps = 1000
n_dim = 2
H = 0.5  # Hurst parameter
T = 2.0  # Total time
dt = T / n_steps
X = cholesky_fbm(n_steps, n_dim, H, T)
X1 = brownian_motion(n_steps, n_dim, T)


reps = 1e5

x_bm1 = zeros(Int(reps), n_steps+1)
x_fbm1 = zeros(Int(reps), n_steps+1)
x_bm2 = zeros(Int(reps), n_steps+1)
x_fbm2 = zeros(Int(reps), n_steps+1)

for i in 1:Int(reps)
    x_bm1[i,:] = brownian_motion(n_steps, n_dim, T)[:, 1]
    x_fbm1[i,:] = cholesky_fbm(n_steps, n_dim, H, T)[:,1]
    x_bm2[i,:] = brownian_motion(n_steps, n_dim, T)[:, 2]
    x_fbm2[i,:] = cholesky_fbm(n_steps, n_dim, H, T)[:, 2]
end

mean_bm1 = mean(x_bm1, dims=1)
mean_fbm1 = mean(x_fbm1, dims=1)
mean_bm2 = mean(x_bm2, dims=1)
mean_fbm2 = mean(x_fbm2, dims=1)
sd_bm1 = std(x_bm1, dims=1)
sd_fbm1 = std(x_fbm1, dims=1)
sd_bm2 = std(x_bm2, dims=1)
sd_fbm2 = std(x_fbm2, dims=1)


p_bm1 = plot(collect(0:dt:T), mean_bm1[:], ribbon=sd_bm1[:], label="BM1", color=:blue, alpha=0.5, title = "Brownian Motion\nx1 dimension")
p_fbm1 = plot(collect(0:dt:T), mean_fbm1[:], ribbon=sd_fbm1[:], label="fBM1 H = 0.5", color=:red, alpha=0.5, title = "Fractional Brownian Motion\nx1 dimension")
p_bm2 = plot(collect(0:dt:T), mean_bm2[:], ribbon=sd_bm2[:], label="BM2", color=:blue, alpha=0.5, title = "Brownian Motion\nx2 dimension")
p_fbm2 = plot(collect(0:dt:T), mean_fbm2[:], ribbon=sd_fbm2[:], label="fBM2 H = 0.5", color=:red, alpha=0.5, title = "Fractional Brownian Motion\nx2 dimension")

plot(p_bm1, p_fbm1, p_bm2, p_fbm2, layout=(2,2), size=(800, 600), legend=:topleft)

bm_1_last = x_bm1[:, end]
fbm_1_last = x_fbm1[:, end]

pval = round(pvalue(ApproximateTwoSampleKSTest(bm_1_last, fbm_1_last)), digits=2)

p1 = histogram(bm_1_last, label="BM1", alpha=0.5, color=:blue, bins=20, title="Histogram of the last value\nBM1 vs fBM1")
histogram!(fbm_1_last, label="fBM1", alpha=0.5, color=:red, bins=20)
title!("Histogram of the last value\nBM1 vs fBM1\np-value = $pval")

bm_2_last = x_bm2[:, end]
fbm_2_last = x_fbm2[:, end]

pval = round(pvalue(ApproximateTwoSampleKSTest(bm_2_last, fbm_2_last)), digits=2)

p2 = histogram(bm_2_last, label="BM2", alpha=0.5, color=:blue, bins=20, title="Histogram of the last value\nBM2 vs fBM2")
histogram!(fbm_2_last, label="fBM2", alpha=0.5, color=:red, bins=20)
title!("Histogram of the last value\nBM2 vs fBM2\np-value = $pval")

plot(p1, p2, layout=(1,2), size=(800, 400), legend=:topleft)


p1 = boxplot(["BM1"], bm_1_last, label="BM1", color=:blue, alpha = 0.5)   
violin!(["BM1"], bm_1_last, label="BM1", alpha=0.7, color=:blue)

p2 = boxplot(["fBM1"], fbm_1_last, label="fBM1", color=:red, alpha = 0.5)
violin!(["fBM1"], fbm_1_last, label="fBM1", alpha=0.7, color=:red)

p3 = boxplot(["BM2"], bm_2_last, label="BM2", color=:blue, alpha = 0.5)
violin!(["BM2"], bm_2_last, label="BM2", alpha=0.7, color=:blue)

p4 = boxplot(["fBM2"], fbm_2_last, label="fBM2", color=:red, alpha = 0.5)
violin!(["fBM2"], fbm_2_last, label="fBM2", alpha=0.7, color=:red)

plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600), legend=:topleft)
