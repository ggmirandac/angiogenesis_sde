import Pkg
    # activate the shared project environment
Pkg.activate(Base.current_project())
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()
using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures, StatsBase, LinearAlgebra,
    Random, Measurements

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
    
    # Generate standard normal samples
    Z = randn(n_steps+1, n_dim)
    
    # Transform to correlated Gaussian process
    X = L * Z
    
    # Scale by time
    X *= dt^H
    
    return t, X
end
    
function brownian_motion(n_steps::Int, n_dim::Int, T::Float64)
    dt = T / n_steps
    t = range(0, T, length=n_steps+1)
    
    distr = Normal(0, dt)
    X = rand(distr, n_steps+1, n_dim)
    
    X = cumsum(X, dims=1)
    return t, X
end


# Example usage
n_steps = 1000
n_dim = 2
H = 0.5  # Hurst parameter
T = 2.0  # Total time

t, X = cholesky_fbm(n_steps, n_dim, H, T)

t1, X1 = brownian_motion(n_steps, n_dim, T)

x_bm1 = zeros(Int(1e5), n_steps+1)
x_fbm1 = zeros(Int(1e5), n_steps+1)
x_bm2 = zeros(Int(1e5), n_steps+1)
x_fbm2 = zeros(Int(1e5), n_steps+1)

for i in 1:Int(1e5)
    x_bm1[i,:] = brownian_motion(n_steps, n_dim, T)[2][:, 1]
    x_fbm1[i,:] = cholesky_fbm(n_steps, n_dim, H, T)[2][:, 1]
    x_bm2[i,:] = brownian_motion(n_steps, n_dim, T)[2][:, 2]
    x_fbm2[i,:] = cholesky_fbm(n_steps, n_dim, H, T)[2][:, 2]
end

mean_bm1 = mean(x_bm1, dims=1)
mean_fbm1 = mean(x_fbm1, dims=1)
mean_bm2 = mean(x_bm2, dims=1)
mean_fbm2 = mean(x_fbm2, dims=1)
sd_bm1 = std(x_bm1, dims=1)
sd_fbm1 = std(x_fbm1, dims=1)
sd_bm2 = std(x_bm2, dims=1)
sd_fbm2 = std(x_fbm2, dims=1)


p_bm1 = plot(t, mean_bm1[:], ribbon=sd_bm1[:], label="BM1", color=:blue, alpha=0.5, title = "Brownian Motion x1 dimension")
p_fbm1 = plot(t, mean_fbm1[:], ribbon=sd_fbm1[:], label="fBM1 H = 0.5", color=:red, alpha=0.5, title = "Fractional Brownian Motion x1 dimension")
p_bm2 = plot(t, mean_bm2[:], ribbon=sd_bm2[:], label="BM2", color=:blue, alpha=0.5, title = "Brownian Motion x2 dimension")
p_fbm2 = plot(t, mean_fbm2[:], ribbon=sd_fbm2[:], label="fBM2 H = 0.5", color=:red, alpha=0.5, title = "Fractional Brownian Motion x2 dimension")

plot(p_bm1, p_fbm1, p_bm2, p_fbm2, layout=(2,2), size=(800, 600), legend=:topleft)
title!("Brownian Motion vs Fractional Brownian Motion")