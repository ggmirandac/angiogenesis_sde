### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ fe2a5cc0-f423-44f1-b456-1e70b1bc5377
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures
end

# ╔═╡ 05efad10-26a4-11ef-187a-4be6f874349d
md"""
# Model
In this notebook we cover the Angiogenesis process stated by the equations of Stokes and Lauffenburger (1991).
The equation that gorvern this process is stated as:
```math
	dv_i(t) = -\beta v_i(t) dt + \sqrt{\alpha} dW_i(t) + \kappa \nabla a \sin \left|\frac{\phi_i}{2}\right| dt
```
Before stating the parameters of the problem, let's rewrite the expression to fit better in the Euler-Maruyama scheme for solving SDE.
```math
	dv_i(t) = \left( -\beta v_i(t) + \kappa \nabla a \sin \left|\frac{\phi_i}{2}\right|  \right) dt + \sqrt{\alpha} dW_i(t) 
```
Variables and parameters:
- W_i is the Weiner process or Brownian motion component of the system.
- The variable $v_i$ represents the two-dimensional speed of the i-th sprout,i.e, $v_i = [{v_i}_x, {v_i}_y]$.
- The parameter $\alpha$ provides the variance $\alpha |t|$ to the system given that the brownian motion is a $\mathcal{N}(0,1)$ brownian motion.
- The parameter $\beta$ represents the decay rate contant of the current velocity
- The parameter $\kappa$ is the chemotactic responsiveness
- The parameter $\nabla a$ is the attractant concentration gradient. This varible can be also seen as a variable in the case of a non-constant attractant concentration gradient.
- The variable $\phi$ represents the angle between the direction the tip is moving that towards to the attractant source. This variable follows the following formulation
```math
\phi_i = \cos^{-1} \left[ \frac{(x_a - x_i)\cos(\theta_i) + (y_a - y_i)\sin(\theta_i)}{ ((x_a - x_i)^2 - (y_a - y_i)^2)^{1/2} }   \right]
```

The solution of this model can be done by using  the Euler-Maruyama method for solving Stochastic Differential equations:

```math
	v_{t_{n+1}} = v_{t_{n}} + \left[ \beta v_{t_n} + \kappa \nabla a \sin\left| \frac{\phi_i}{2}\right| \right] (t_{n+1} - t_n) + \sqrt{\alpha} (W_{t_{n+1}} + W_{t_n})
```
Then from this, the positions can be defined as:
```math
\begin{align}
	x_{t_{n+1}} = v_{t_{n+1}, x} (t_{n+1} - t_n) + x_{t_{n}}\\
	y_{t_{n+1}} = v_{t_{n+1}, y} (t_{n+1} - t_n) + y_{t_{n}}
\end{align}
```

# References
- Stokes, C. L., & Lauffenburger, D. A. (1991). Analysis of the roles of microvessel endothelial cell random motility and chemotaxis in angiogenesis. Journal of Theoretical Biology, 152(3), 377–403. https://doi.org/10.1016/S0022-5193(05)80201-2

"""



# ╔═╡ bb190ac7-4a2a-45a7-a7b6-0976819225e8
"""
	sphere_gradient(xa, ya, xi, yi, a0, R)
Returns a constant concentration gradient at every (x,y)
# Arguments
- `xi, yi`: Position of the tip of the capillary
- `xa, ya`: Position of the source
- `a0`: Initial concentration
- `R`: Radious of source
"""
function sphere_gradient(xa, ya, xi, yi, a0, R)
	distance = sqrt((xa-xi)^2 + (ya-yi)^2)
	
	if distance > R
		delta_a = [a0*R/(distance), a0*R/(distance)]
	else 
		delta_a = "stop"
	end
	return delta_a
end

# ╔═╡ 0b90fbdc-b1b6-4210-b253-da83c2b3e439
function divergence_gradient(xi, yi, force)
	if yi == 0
		return [0;0]
	elseif yi < 0 
		return [0;-force]
	else
		return [0;force]
	end
end

# ╔═╡ 6bffff2d-0b84-483b-be56-ce1d9e6068cd
"""
	define_next_x(x_t, v_t1, dt)
Defines the state x\\_(t+1) given the velocity v\\_(t+1). 
# Arguments
- x_t: position at time t
- v_t: velocity at time v_t
- dt: time step length
"""
function define_next_x(x_t, v_t, dt)
	x_t1 = x_t + v_t * dt
	return x_t1
end

# ╔═╡ 5f979125-77df-41bb-99e2-fcedd0598db3
"""
	calculate_phi(xi, yi, xia, yia, theta)
Calculates the phi angle given the equation given by the Stokes and Lauffenburger formulation
# Arguments
- xi: x-coordinate of the tip of the sprout
- yi: y-coordinate of the tip of the sprout
- xia: x-coordinate of the chemoattractant source
- yia: y-coordinate of the chemoattractant source
- theta: Direction of movement of the tip of the sprout
"""
function calculate_phi(xi, yi, xia, yia, theta)
	numerator = (xia - xi) * cos(theta) + (yia - yi) * sin(theta)
	denominator = ((xia - xi)^2  + (yia - yi)^2)^(1/2)
	phi = acos(round(numerator / denominator; digits= 10))
	return phi
end 

# ╔═╡ b3f71284-feb2-47e9-94dd-60e4d51b382f
"""
	simulate_sprout(x0, nabla_a, beta, alpha, kappa; dt = 1, v0 = [0,0], 
							n_steps = 100)
Simulates the sprout given the Stokes and Lauffenburger model.
# Arguments
- x0: x and y position at the t = 0
- nabla_a: attractant concentration gradient
- beta: decay rate constant
- kappa: chemotactic responsiveness
- dt: delta t of the model, i.e., the temporal length of steps
- v0: Initial velocity of the tip of the sprout
- n_steps: amount of steps to simulate
# References
	Stokes, C. L., & Lauffenburger, D. A. (1991). Analysis of the roles of microvessel endothelial cell random motility and chemotaxis in angiogenesis. Journal of Theoretical Biology, 152(3), 377–403. https://doi.org/10.1016/S0022-5193(05)80201-2

"""
function simulate_sprout(x1, nabla_a, beta, alpha, kappa, xa, ya; T = 100, N = 300, R = 2, v1 = [0;0], a0 = 10^-10, Radious = 1)
	# Euler-Maruyama simulations steps
	dt = T/N
	DT = R*dt
	L = Int(N/R)
	# dataframes to store paths
	df_x = DataFrames.DataFrame(time = Any[], x = Float64[], y = Float64[])
	df_v = DataFrames.DataFrame(time = Any[], x = Float64[], y = Float64[])

	# Normal for random motion
	normal = Distributions.Normal(0, dt) # Note that a brownian motion has variance t_(n+1) - t_n = dt
	dW = rand(normal, (2, N))
	
	## initialization of parameters:
	phi = 0 # Initialization of the phi 
	theta = 0 # Initialization of theta
	v_vect = Vector{Float64}[v1] # Vector to contain the sprout velocity over time
	x_vect = Vector{Float64}[x1] # Vector to contain the sprout position over time
	
	push!(df_x, [0, x_vect[1][1], x_vect[1][2]])
	push!(df_v, [0, v_vect[1][1], x_vect[1][2]])
	for step = 2:L
		time = (step-1) * DT
		# deterministic step
		v_i_1 = v_vect[step-1]
		x_i_1 = x_vect[step-1]
		# deterministic
		gradiente = nabla_a(x_i_1[1], x_i_1[2], a0)
		if gradiente == "stop"
			return df_x, df_v
		end
		det_vi = ( - beta .*v_i_1) .* DT 
		#bias

		bias_vi =( kappa * gradiente * sin(phi/2) ) * DT
		# random
		rand_vi = vec(sqrt(alpha) .* sum(dW[:, R*(step-1)+1:R*step];dims = 2))
		# sum to v
		v_i = v_i_1 +  det_vi + rand_vi + bias_vi

		x_i = x_i_1 + v_i .* DT
		
		push!(v_vect, v_i)
		push!(x_vect, x_i)
		
		# Now we calculate the phi and theta for the next step
		theta = atan(
			(x_vect[step][2] - x_vect[step-1][2]) /(x_vect[step][1] - x_vect[step-1][1])
		)
		
		phi = calculate_phi(x_vect[step][1], x_vect[step][2], xa , ya, theta)
		push!(df_x, [time, x_vect[step][1], x_vect[step][2]])
		push!(df_v, [time, v_vect[step][1], v_vect[step][2]])
	end
	return df_x, df_v
end

# ╔═╡ e97e4b6f-d90b-4885-8351-bce5762e623b
begin
	# Run simulation
	# Parameters
	x1 = [0;0]
	nabla_a = divergence_gradient
 	beta = 0.99 # h^-1
    alpha = 1900 # µm^2 h^-3
	# Now we generate the kappa parameter using the derivation of S and L
    # using the delta parameter
    delta = 3 # adimentional chemoatractant
    a0 = 1 # initial concentration of the chemoatractant
	Radious = 1
    kappa = 10 # chemotactic responsiveness 
	xa = 0
	ya = 100

	#temporal for simul
	T = 200 # time from 0 to T
	N = 600 # there are going to be 300 steps
	R = 2;



	Plots.plot(x[:,"x"], x[:,"y"])
	
end

# ╔═╡ d78310bd-f448-4ab5-b6d1-0e2034e29e40
begin
	n_s = 100
	Plots.plot(title = "Simulation of Sprout growing, n = $n_s", dpi = 600)

	for i = 1:n_s
			xv = simulate_sprout(x1, nabla_a, beta, alpha, kappa, xa, ya; T = 100, N = 300, R = 2, v1 = [0;0], a0 = a0, Radious = Radious)
		x = xv[1]

		Plots.plot!(x[:,"x"], x[:,"y"], label = "")
		Plots.scatter!([x[end,"x"]],[x[end,"y"]], label="", markershape =  :xcross, markersize = 6,color = "red", markerstrokewidth = 4)
		Plots.scatter!([x[1,"x"]],[x[1,"y"]],  label="", markershape = :star5, markersize = 6)
	end
	Plots.xlabel!("X-dimension")
	Plots.ylabel!("Y-dimension")
		
end

# ╔═╡ Cell order:
# ╠═05efad10-26a4-11ef-187a-4be6f874349d
# ╠═fe2a5cc0-f423-44f1-b456-1e70b1bc5377
# ╠═bb190ac7-4a2a-45a7-a7b6-0976819225e8
# ╠═0b90fbdc-b1b6-4210-b253-da83c2b3e439
# ╠═6bffff2d-0b84-483b-be56-ce1d9e6068cd
# ╠═5f979125-77df-41bb-99e2-fcedd0598db3
# ╠═b3f71284-feb2-47e9-94dd-60e4d51b382f
# ╠═e97e4b6f-d90b-4885-8351-bce5762e623b
# ╠═d78310bd-f448-4ab5-b6d1-0e2034e29e40
