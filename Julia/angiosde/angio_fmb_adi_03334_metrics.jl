import Pkg
    # activate the shared project environment
Pkg.activate(Base.current_project())
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()
using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures, StatsBase, LinearAlgebra,
    Random, Measurements, StatsPlots, HypothesisTests, StatsPlots, Pickle, JLD

"""
    divergent_gradient(x; force = 2)
Calculates the gradient of the divergence of the gradient of the chemoattractant source
# Arguments
- x: position of the sprout
- force: the force of the gradient
"""
function divergent_gradient( x ; xforce = 2, yforce = 2)
    if x[1] > 0 && x[2] > 0
        delta_a = [xforce, yforce]
    elseif x[1] < 0 && x[2] < 0
        delta_a = [-xforce, -yforce]
    elseif x[1] > 0 && x[2] < 0
        delta_a = [xforce, -yforce]
    elseif x[1] < 0 && x[2] > 0
        delta_a = [-xforce, yforce]
    else 
        delta_a = [0,0]
    end
end

""" 
sphere_gradient sphere_gradient(x, xa, a0, R)

Returns a constant concentration gradient at every (x,y)
# Arguments
- `x`: position of the sprout
- `xa`: chemoattractant source
- `a0`: Initial concentration
- `R`: Radious of source

"""

function sphere_gradient(x, xa, a0, R)
    distance = sqrt((xa[1]-x[1])^2 + (xa[2]-x[2])^2)
    
    if distance > R
        delta_a = [a0*R/(distance), a0*R/(distance)]
    else 
        delta_a = "stop"
    end
    return delta_a
end

"""
constant_gradient(x, xa, a0)
Returns a constant concentration gradient at every (x,y)
# Arguments
- `x`: position of the sprout
- `xa`: chemoattractant source
- `a0`: Initial concentration
"""
function upwards_ct_gradient(x, xa, a0; force  = 2)
    return [0, force] # up wards force of movement
    
end

# Auxiliary functions
"""
    calculate_phi(xi, yi, xia, yia, theta)
Calculates the phi angle given the equation given by the Stokes and Lauffenburger formulation
# Arguments
- xi: x-coordinate of the tip of the sprout
- yi: y-coordinate of the tip of the sprout
- xia: x-coordinate of the chemoattractant source
- yia: y-coordinate of the chemoattractant source
- theta: angle of the gradient
"""

function calculate_phi(xi,xa, theta)
    #(xi, " ", yi, " ", xia, " ", yia, " ", theta, "\n")
    numerator = (xa[1] - xi[1]) * cos(theta) + (xa[2] - xi[2]) * sin(theta)
    denominator = ((xa[1] - xi[1])^2  + (xa[2] - xi[2])^2)^(1/2)
    phi = acos(round(numerator / denominator; digits = 10))
    return phi
end

"""
    calculate_theta(xi, yi, xia, yia)   
Calculates the theta angle given the equation given by the Stokes and Lauffenburger formulation
# Arguments
- xi: x-coordinate of the tip of the sprout
- yi: y-coordinate of the tip of the sprout
- xia: x-coordinate of the chemoattractant source
- yia: y-coordinate of the chemoattractant source
"""
function calculate_theta(xi, xi_1)
    vector_x_pos = [1,0]
    vector_distance = [xi[1] - xi_1[1], xi[2] - xi_1[2]]
    theta = acos(dot(vector_x_pos, vector_distance)/(norm(vector_x_pos)*norm(vector_distance)))
    return theta
end

"""
    brownian_motion(n_steps::Int, n_dim::Int, T::Float64)
Generates a Brownian motion
# Arguments
- `n_steps`: number of steps
- `n_dim`: number of dimensions
- `T`: total time
"""
function brownian_motion(n_steps::Int64, n_dim::Int64, Δt::Float64)
    
    distr = Normal(0,sqrt(Δt))
    gn = rand(distr, n_steps, n_dim)
    return gn
end    

""" 
    rho(n, H)
Calculates the autocorrelation function of the fractional brownian motion
# Arguments
- `n`: the distance between the two points
- `H`: Hurst parameter
"""
function rho(n, H)
    ρn = 0.5 * (abs(n+1)^(2*H) + abs(n-1)^(2*H) - 2 * abs(n)^(2*H))
    return ρn
end

"""
    cov_matrix(n_steps, H)
Calculates the covariance matrix of the fractional brownian motion

# Arguments
- `n_steps`: number of steps of the motion
- `H`: Hurst parameter
"""
function cov_matrix(n_steps, H)
    C = zeros(n_steps, n_steps)
    for i in 1:n_steps
        for j in 1:n_steps
            C[i, j] = rho(abs(i-j), H)
        end
    end
    return C
end


"""
    cholesky_fbm(n_steps::Int, n_dim::Int, H::Float64, Δt::Float64, L::LowerTriangular{Float64, Matrix{Float64}})
Calculate the fractional brownian motion using the cholesky decomposition method
# Arguments
- `n_steps`: number of steps of the motion
- `n_dim`: number of dimensions of the motion
- `H`: Hurst parameter
- `Δt`: time step
- `L`: Lower triangular matrix has to be calculated using the `cov_matrix`. For speed reasons it is better to calculate it once and pass it as an argument
"""
function  cholesky_fgm(n_steps::Int, n_dim::Int, H::Float64, Δt::Float64, L::LowerTriangular{Float64, Matrix{Float64}})
    normal = Normal(0, 1)
    normal = Normal(0, 1)
    Z = rand(normal, n_steps, n_dim)
    fgn = L * Z
    fgn *= Δt ^ H
    return fgn
end


"""
    wrapper_div_gradient(x)
Wrapper function for the divergence gradient
# Arguments    
- x: position of the sprout
- xa: chemoattractant source
"""
function wrapper_div_gradient(x, xa, a0)
    
    x_force = 10^(-1.5)
    y_force = 10^(-1.5)
    return divergent_gradient(x; xforce = x_force, yforce = y_force)
end    
"""
    wrapper_shpere_gradient(x)
Wrapper function for the sphere gradient
# Arguments    
- x: position of the sprout
- xa: chemoattractant source
"""
function wrapper_shpere_gradient(x, xa, a0)
    R = 1
    return sphere_gradient(x, xa, a0, R)
end    

"""
    wrapper_upwards_ct_gradient(x)
Wrapper function for the upwards constant gradient
# Arguments    
- x: position of the sprout
- xa: chemoattractant source
"""
function wrapper_upwards_ct_gradient(x, xa, a0)
    force = 10^(-0.5)
    return upwards_ct_gradient(x, xa, a0; force = force)
end    


# Simulation
"""
    simulate_sprout(x1, δ, w_gradient_function, Δτ; n_steps, v1, xa)
Simulates the sprout in adimentioanl units to get a better understanding of the chemotactic responsiveness
# Arguments
- `x1`: initial position and velocity
- `δ`: adimentioanl chemotactic responsiveness
- `w_gradient_function`: wrapper for the function that calculates the gradient
- `Δτ`: time step
- `n_steps`: the number of steps to simulate this is sum_n_steps Δτ until
- `v1`: initial velocity
- `xa`: chemoattractant source
"""
function simulate_sprout(
    x1, # initial position and velocity
    δ, # adimentioanl chemotactic responsiveness
    w_gradient_function,
    Δτ # time step
    ;
    n_steps = 1000, # the number of steps to simulate this is sum_n_steps Δτ until
    v1 = [0.0, 0.0], # initial velocity,
    xa = [0.,0.],
    a0 = 10e-10,
    div_grad_source = false,
    const_up_grad = false,
    random_source = "BM",
    H = 0.5,
    L = nothing
)
    x_history = zeros(2, n_steps)
    v_history = zeros(2, n_steps)
    if random_source == "BM"
        gn = brownian_motion(n_steps, 2, Δτ)
    elseif random_source == "fBM"
        fgn = cholesky_fgm(n_steps, 2, H, Δτ, L)
    end
    x = x1
    v = v1
    θ = 0
    if const_up_grad
        xa = [0, 100]
    end

    φ = calculate_phi(x, xa, θ)
    for i = 1:n_steps
        x_history[:, i] .= x
        v_history[:, i] .= v
        if random_source == "BM"
            weiner_proc = gn[i, :] 
        elseif random_source == "fBM"
            weiner_proc = fgn[i, :] 
        end        
        # calculate the gradient
        gradient = w_gradient_function(x, xa, a0)
        # if div_grad_source == true then generate 4 different gradients sources for each quadrant
        if div_grad_source
            if x[1] > 0 && x[2] > 0
                xa = [100, 100]
            elseif x[1] < 0 && x[2] < 0
                xa = [-100, -100]
            elseif x[1] > 0 && x[2] < 0
                xa = [100, -100]
            elseif x[1] < 0 && x[2] > 0
                xa = [-100, 100]
            else 
                xa = [0,0]
            end
        end
        # calculate the next velocity... 
        #  resitance of change in velucity, randomness, bias
        #v = v .+ (- v * Δτ) .+ (weiner_proc) .+ (δ .* gradient .* sin(φ/2) * Δτ)
        v_res = - v * Δτ
        v_rand = weiner_proc 
        v_bias = δ .* gradient .* sin((φ/2)) * Δτ
        v = v .+ v_res .+ v_rand .+ v_bias
        x = x .+ v .* Δτ
        θ = calculate_theta(x, x_history[:, i])
        φ = calculate_phi(x, xa, θ)
    end 
    return x_history, v_history
end


# initial position and velocity
x1 = [0.0, 0.0]
v1 = [0.0, 0.0]
# chemoattractant source
xa = [0., 0.]
δ = 1.2 # > 0  implies there is more chemotactic drift towards the source, < 0 more random
Δτ = .01
n_steps = Int(1e4)

wrp_grad = wrapper_upwards_ct_gradient


# simulate the sprout
Plot_traj = true
plot(title = "Trajectory of the sprout", xlabel = "x", ylabel = "y")
n_reps = 100_000
L = cholesky(cov_matrix(n_steps, 0.3334333333333333)).L
# local scope for plotting
# local scope for plotting

# let 
#     global bm_final_x = zeros(n_reps, 2)
#     global bm_distance = zeros(n_reps, 1)
#     for i = 1:n_reps

#         x_plot, v_plot = simulate_sprout(
#             x1, # initial position and velocity
#             δ, # adimentioanl chemotactic responsiveness
#             wrp_grad,
#             Δτ; # time step

#             n_steps = n_steps, # the number of steps to simulate this is sum_n_steps Δτ until
#             v1 = v1, # initial velocity,
#             xa = [1.,1.],  
#             a0 = 10e-8,
#             div_grad_source = false,
#             random_source = "BM",
#             H = 0.5,
#             L = L
#         )
#         final_position = x_plot[:, end]
#         bm_final_x[i, :] = transpose(final_position)
#         bm_distance[i] = sqrt(final_position[1]^2 + final_position[2]^2)
#     end
# end



let 
   
    global fbm_final_x = zeros(n_reps, 2)
    global distance_fbm = zeros(n_reps, 1)
    for i = 1:n_reps

        x_plot, v_plot = simulate_sprout(
            x1, # initial position and velocity
            δ, # adimentioanl chemotactic responsiveness
            wrp_grad,
            Δτ; # time step
            n_steps = n_steps, # the number of steps to simulate this is sum_n_steps Δτ until
            v1 = v1, # initial velocity,
            xa = [1.,1.],  
            a0 = 10e-8,
            div_grad_source = false,
            random_source = "fBM",
            H = 0.3334333333333333,
            L = L
        )
        final_pos = x_plot[:, end]
        fbm_final_x[i, :] = transpose(final_pos)
        distance_fbm[i] = sqrt(final_pos[1]^2 + final_pos[2]^2)
    end
end

## Pickling

# JLD.save("jlds/brownian_motion_final_pos.jld", "bm_final_x", bm_final_x)
# JLD.save("jlds/brownian_motion_distance.jld", "bm_distance", bm_distance)
JLD.save("jlds/fbm_final_pos03334.jld", "fbm_final_x", fbm_final_x)
JLD.save("jlds/fbm_distance03334.jld", "distance_fbm", distance_fbm)





# pval_1 = round(pvalue(ApproximateTwoSampleKSTest(bm_final_x[:,1], fbm_final_x[:, 1])), digits=2)
# pval_2 = round(pvalue(ApproximateTwoSampleKSTest(bm_final_x[:,2], fbm_final_x[:, 2])), digits=2)


# plot1 = violin(["coord_x"],bm_final_x[:, 1], side = :left, label = "BM", color = :dodgerblue,
# legend = :topleft)
# violin!(["coord_x"],fbm_final_x[:, 1], side = :right, label = "fBM - H = 0.5", color = :lightslateblue,
# legend = :topleft)
# violin!(["coord_y"], bm_final_x[:, 2], side = :left, label = "BM", color = :orange,
# legend = :topright)
# violin!(["coord_y"],fbm_final_x[:, 2], side = :right, label = "fBM - H = 0.5", color = :gold)
# title!("Final position of the sprout. \nKSTest p-value: $pval_1, $pval_2", legend = :topleft, size = (800, 600))
# savefig(plot1,"angio_fmb_adi_05_position.svg")



# pdis = round(pvalue(ApproximateTwoSampleKSTest(bm_distance[:], distance_fbm[:])), digits=3)
# plot3 = boxplot(["fbm"],distance_fbm[:], side = :left, label = "fBM - H = 0.5", color = :dodgerblue)
# violin!(["fbm"], distance_fbm[:], label = "", alpha = 0.5, color = :dodgerblue)
# boxplot!(["bm"],bm_distance[:], side = :right, label = "BM", color = :lightslateblue)
# violin!(["bm"],bm_distance[:], label = "", color = :lightslateblue, alpha = 0.5)
# title!( "Final distance from the origin. \nKSTest p-value: $pdis", legend = :topright, size = (800, 600))
# savefig(plot3,"angio_fmb_adi_05_metrics.svg")

