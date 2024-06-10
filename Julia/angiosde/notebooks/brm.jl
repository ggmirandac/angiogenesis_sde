### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ b4cd95a6-1f90-11ef-2a0d-77e3a547aecf
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Plots, Distributions
end

# ╔═╡ d121854b-b4c0-4e01-9b7c-c00af5944be1
md"""
# Brownian Motion
"""

# ╔═╡ 8bf5a0e4-0593-43d7-ac7d-3cc4b9c97d2f
md"""
The Brownian motion is defined as:
``
$
dX_{t} = dW_t
$
``
By using the Euler-Maruyama Method we have:
``
$
X_{t_{n+1}} = X_{t_n} + (W_{t_{n+1}} - W_{t_n})
$
``
``
$
W_t - W_s \sim \mathcal{N}(0, (t-s))
$
``

For example, in Julia we have the Distributions package that helps us to sample from a Normal distribution in the following way
```julia
normal_dist = Distributions.Normal(mu, std)
normal_sample = rand(normal_dist, 100) # sample 100 iid numbers from the N(mu, std) sample
```
"""

# ╔═╡ 26f55506-67c1-45f3-8442-fbe2bcf72196
begin 
	mu = 0
	std = 1
	norm = Distributions.Normal(mu, std)
	x = rand(norm, 10000)
	a = Plots.histogram(x, title="Normal Distribution with `Distributions`", label="x")
end


# ╔═╡ 7e6f1a25-204a-4eea-9da8-0e88b78a54a2
md"""
We can implement the Euler-Maruyama method in Julia in a straightforward way given that it is an iterative formula. Hence, what we do is.
1. Know the number of steps
2. Assign the $dt = t-s$
3. Assign $x_0$. The default value is 0 in this implementation.

Note that we do:
```math
X_t = X_{t-1} + N_t \cdot \sqrt{dt}, \text{with } N_t \sim \mathcal{N}(0, 1)
```
This is because:
```math
Z = X\cdot \sigma  + \mu
```
```math
Z\sim\mathcal{N}(\mu, \sigma) \And \;X\sim\mathcal{N}(0,1)
```
"""

# ╔═╡ 3bbaf57c-995b-4fa0-91c1-63096573a46c
begin 
	function bm(n_steps, dt, x0=0)
		v = zeros(n_steps, 1)
		v[1] = x0
		normal_dist = Distributions.Normal(0, 1)
		normal_list = rand(normal_dist, n_steps)
		for i = 2:n_steps
			v[i] = v[i-1] + normal_list[i] * sqrt(dt)
		end
		return v
	end 
end


# ╔═╡ a8392e63-9f36-457f-8fbe-fe0b167aa5fa
begin 
	n_steps = 1000
	dt = 0.1
	tf = n_steps * dt - dt
	x0 = 0 
	t = 0:dt:tf
	n_paths = 100
	ploti = Plots.plot(title = "Brownian motions n = $n_paths", dpi = 600)	
	for i = 1:n_paths
		bm_v = bm(n_steps, dt, x0)
		Plots.plot!(t, bm_v, label = "")
	end
	Plots.xlabel!("time")
	Plots.ylabel!("Path")
end

# ╔═╡ 2fbd1939-7405-4bd9-94b5-3dc81df6927d
md"""
### Stochastic Differential Equations
We can define an SDE as
```math
dX_t = a(t, X_t)dt + b(t, X_t) dW_t
```
And can also be simulated with the Euler-Maruyama method
```math
X_{t_{n+1}} = X_{t_{n}} + a(t_n, X_{t_n}) (t_{n+1} - t_n) + b(t_n, X_{t_n}) (W_{t_{n+1}} - W_{t_n})
```
One example of these cases are, for example, an Ornstein-Unlenback process, which is modeled as:
```math
dX = \theta(\mu - X_t) dt + \sigma dW
```
By applying the Euler-Maruyama Method we have:
```math
X_{t} = X_{t-1} + \theta(\mu - X_t) \delta t + \sigma (W_t - W_{t-1}) 
```
Given a $N_i \sim\mathcal{N}(0,1)$ we have:
```math
X_{t} = X_{t-1} + \theta(\mu - X_t) \delta t + \sigma N_t \sqrt{\delta t}
```
"""

# ╔═╡ 2b619820-c748-4559-9069-7324c3de3e9a
function OUP(n_steps, dt, x_0; theta = 0.7, mu = 1.5, sigma = 0.06)
	v = zeros(n_steps, 1)
	norm = Distributions.Normal(0,1)
	sample = rand(norm, n_steps)
	v[1] = x0
	for i = 2:n_steps
		v[i] = v[i-1]+theta*(mu - v[i-1]) * dt + sigma * sample[i] * sqrt(dt)
	end
	return v
end

# ╔═╡ 24104a15-5ae3-468e-bef5-3afff3d2bf0a
begin
	n_steps2 = 7000
	dt2 = 0.001
	x_02 = 0
	tf2 = n_steps2 * dt2 - dt2
	t2 = 0:dt2:tf2
	
	
	n_reps = 10
	p = Plots.plot(title = "Ornstein - Unlenback process", dpi = 600)
	for i = 1:n_reps
		vi = OUP(n_steps2, dt2, x_02)
		p = Plots.plot!(t2, vi, label = "")
	end
	Plots.plot(p)
	
end

# ╔═╡ 198364c5-6833-4b60-84d0-420ba3f2811a


# ╔═╡ Cell order:
# ╠═b4cd95a6-1f90-11ef-2a0d-77e3a547aecf
# ╟─d121854b-b4c0-4e01-9b7c-c00af5944be1
# ╟─8bf5a0e4-0593-43d7-ac7d-3cc4b9c97d2f
# ╠═26f55506-67c1-45f3-8442-fbe2bcf72196
# ╟─7e6f1a25-204a-4eea-9da8-0e88b78a54a2
# ╠═3bbaf57c-995b-4fa0-91c1-63096573a46c
# ╠═a8392e63-9f36-457f-8fbe-fe0b167aa5fa
# ╠═2fbd1939-7405-4bd9-94b5-3dc81df6927d
# ╠═2b619820-c748-4559-9069-7324c3de3e9a
# ╠═24104a15-5ae3-468e-bef5-3afff3d2bf0a
# ╠═198364c5-6833-4b60-84d0-420ba3f2811a
