import Pkg
    # activate the shared project environment
Pkg.activate(Base.current_project())
# instantiate, i.e. make sure that all packages are downloaded
Pkg.instantiate()
using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures, StatsBase, LinearAlgebra,
    Random, Measurements

# gradients

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
    vector_x_pos = [0,1]
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
function brownian_motion(n_steps::Int64, n_dim::Int64, T::Float64)
    dt = T / n_steps
    t = range(0, T, length=n_steps+1)
    
    distr = Normal(0, dt)
    X = rand(distr, n_steps+1, n_dim)
    
    X = cumsum(X, dims=1)
    return t, X
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
    const_up_grad = false
)
    x_history = zeros(2, n_steps)
    v_history = zeros(2, n_steps)
    T = Float64(Δτ * n_steps)
    t, bm = brownian_motion(n_steps, 2, T)

    x = x1
    v = v1
    θ = 0
    if const_up_grad
        xa = [0, 100]
    end

    φ = calculate_phi(x, xa, θ)
    for i in 1:n_steps
        x_history[:, i] .= x
        v_history[:, i] .= v
        weiner_proc = bm[i+1, :] - bm[i, :]
        
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
    force = 10^(-1.)
    return upwards_ct_gradient(x, xa, a0; force = force)
end
# body

# initial position and velocity
x1 = [0.0, 0.0]
v1 = [0.0, 0.0]
# chemoattractant source
xa = [0., 0.]
δ = 1.2 # > 0  implies there is more chemotactic drift towards the source, < 0 more random
Δτ = .1
n_steps = Int(1e4)

wrp_grad = wrapper_upwards_ct_gradient


# simulate the sprout
Plot_traj = true
Plots.plot(title = "Trajectory of the sprout", xlabel = "x", ylabel = "y")
n_reps = 100
# local scope for plotting
let 
    if Plot_traj == true
        max_y = 0
        max_x = 0
        min_x = 0 
        min_y = 0
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
                div_grad_source = false
            )
            Plots.plot!(x_plot[1, :], x_plot[2, :], label="")
            Plots.scatter!([x_plot[1, end]],[ x_plot[2, end]], markersize = 5, color = "red", label="")
            if maximum(x_plot[2, :]) > max_y
                max_y = maximum(x_plot[2, :])
            end
            if minimum(x_plot[2, :]) < min_y
                min_y = minimum(x_plot[2, :])
            end
            if maximum(x_plot[1, :]) > max_x
                max_x = maximum(x_plot[1, :])
            end
            if minimum(x_plot[1, :]) < min_x
                min_x = minimum(x_plot[1, :])
            end
        end
        x_lims = range(min_x, max_x, length = 100)
        y_lims = range(min_y, max_y, length = 100)
        Plots.plot!(x_lims, y_lims.*0, color = "grey", label="", linestyle = :dot, width = 2)
        Plots.plot!(x_lims.*0,y_lims, color = "grey", label="", linestyle = :dot, width = 2)
        Plots.xlabel!(L"x [a.u.]")
        Plots.ylabel!(L"y [a.u.]")
    end
end
# # analysis of sprout

# final_y, final_x = [], []
nre = 100
x_lists, y_lists = zeros(nre, n_steps), zeros(nre, n_steps)
vx_lists, vy_lists = zeros(nre, n_steps), zeros(nre, n_steps)
for i = 1:nre
    x_plot, v_plot = simulate_sprout(
        x1, # initial position and velocity
        δ, # adimentioanl chemotactic responsiveness
        wrp_grad,
        Δτ; # time step
        n_steps = n_steps, # the number of steps to simulate this is sum_n_steps Δτ until
        v1 = v1, # initial velocity,
        xa = [1.,1.],  
        a0 = 10e-8,
        div_grad_source = false
        )
    x_lists[i, :] = x_plot[1, :]
    y_lists[i, :] = x_plot[2, :]
    vx_lists[i, :] = v_plot[1, :]
    vy_lists[i, :] = v_plot[2, :]
        
end
    
    # Plots.plot(title = "Histogram of the final position of the sprout", xlabel = "y - final position", ylabel = "count")
    # positives = final_y[final_y .> 0]
    # negatives = final_y[final_y .< 0]
    # Plots.histogram!(final_y, bins = 50, label = "y>0: $(length(positives))", alpha = 0.5)
plot_stats = true
if plot_stats == true
    
    mean_x = mean(x_lists; dims = 1)[1,:]
    mean_y = mean(y_lists; dims = 1)[1,:]
    std_x = std(x_lists; dims = 1)[1,:]
    std_y = std(y_lists; dims = 1)[1,:]
    
    Plots.plot(title = "Mean and standard deviation of the sprout", xlabel = "x", ylabel = "y")
    Plots.plot!(mean_x .+ std_x, mean_y .+ std_y, label = "Mean - std", color = "red")
    Plots.plot!(mean_x .- std_x, mean_y .- std_y, label = "", color = "blue")
    Plots.plot!(mean_x, mean_y, label = "Mean", color = "green")
end
# autocorr  


plot_autocorr = true
if plot_autocorr == true
    Plots.plot(title = "Autocorrelation of the sprout")
    for i = 1:100
        Plots.plot!(StatsBase.autocor(vx_lists[i,:], collect(1:10:length(x_lists[i,:])); demean = true), label = "")
    end
    Plots.title!("Autocorrelation of the sprout")
end