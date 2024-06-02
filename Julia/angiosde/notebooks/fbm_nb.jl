### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ accbc1e0-0f92-4fba-8ac5-5bda66b67e21
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
	using Plots, Distributions, Brownian, DataFrames, LaTeXStrings, Measures
end

# ╔═╡ 1d0afe6c-2070-11ef-390b-87ebed423db6
md"""
# Fractional Brownian Motion
The Fractional Brownian Motion is a relaxation of the Brownian motion. The Brownian motion is defined as a Markow Gaussian Process, i.e., there is no memory of the process behind the last step and the linear combination of the samples has a Gaussian distribution. (Dieker, 2004).

The Fractional Brownian Motion (fBm) is defined by the following equation:
```math
B_H(t) = \frac{1}{\Gamma(H + 1/2)}\left( \int_{-\infty}^0 [(t-s)^{H-1/2} - (-s)^{H-1/2}]dB(s) + \int_0^t (t-s)^{H-1/2} dB(s) \right)
```
Where H is defined as the Hurst parameter.

Then the fBm  [Also defined as $B_H$] is defined only as a Gaussian process that has the following properties:
1. Stationary Increments
2. For $t\ge0$: $B_H(0) = 0$ and $\mathbb{E}[B_H(t)] = 0$
3. For $t\ge0$: $\mathbb{E}[B_H(t)^2] = t^{2H}$
4. Each realization has a Gaussian Distribution, i.e. $B_H(t) \sim \mathcal{N}(0,1)$

Another fact that is important to note is that the realizations in each time step have a non-0 covariance matrix (except for H = 1/2 which is a standard Bm), this covariance matrix is defined as:
```math
\rho(s,t) = \mathbb{E}[B_H(s)B_H(t)] = \frac{1}{2}\{ t^{2H} + s^{2H} - (t-s)^{2H} \}
```
# Fractional Gaussian Noise

It is crucial to define also the Fractional Gaussian Noise (fGn) to calculate the fBm. The Fractional Gaussian Noise is defined as:
```math
X_k = B_H(k+1) - B_H(h)
```
Which is used to calculate the $B_H(k+1)$ in an iterative way.

To simulate this process we can use [Brownian.jl](https://github.com/uow-statistics/Brownian.jl) package.

In this case, we are going to use an external package given that simulating a fBm needs to follow algorithms like **Hoskin**, **Cholesky**, and **Davies and Harte** methods.
"""

# ╔═╡ 119c7c5e-aafd-464f-a585-a60995f157e9
md"""
Here we have a sampling from the fBm by changing the Hurst parameters.
"""

# ╔═╡ 3d4c5a6a-d811-49b3-980e-5b42551261f9
begin
	# Sampling 
	t0 = 0
	dt = 1/2^10
	tf = 20
	time_ = t0:dt:tf
	dfbm = DataFrame(H = Any[], ts = Any[])
	for i in 0.1:0.2:0.9
		for _ = 1:10
			fbm_i = FBM(time_, i)
			sample = rand(fbm_i)
			push!(dfbm,[i, sample])
		end
	end
	h_01 = subset(dfbm, :H => h -> h .== 0.1)
	h_03 = subset(dfbm, :H => h -> h .== 0.3)
	h_05 = subset(dfbm, :H => h -> h .== 0.5)
	h_07 = subset(dfbm, :H => h -> h .== 0.7)
	h_09 = subset(dfbm, :H => h -> h .== 0.9)
	all_tg = vcat(h_01[1:1, :], h_03[1:1, :], h_05[1:1, :], h_07[1:1, :], h_09[1:1, :])
end

# ╔═╡ 49cdd21e-8a7f-4fd8-ab09-6442f3ee4af8
md"""
This defines the different fBm with different Hurst parameters
"""

# ╔═╡ 8b0ba491-3356-404c-a746-fe71a6d23e89
begin
	# Ploting
	p01 = Plots.plot(time_, h_01.ts, label = "", 
		lw = 3, 
		xlabel = "Time", xguidefontsize = 14,
		ylabel = L"B_H(t)", yguidefontsize = 14,
		tickfontsize = 12,
		title = "H = 0.1",titlefontsize = 16)
	p03 = Plots.plot(time_, h_03.ts, label = "", 
		lw = 3,
		xlabel = "Time", xguidefontsize = 14, 
		ylabel = L"B_H(t)", yguidefontsize = 14,
		tickfontsize = 12,
		title = "H = 0.3", titlefontsize = 16)
	p05 = Plots.plot(time_, h_05.ts, label = "", 
		lw = 3,
		xlabel = "Time", xguidefontsize = 14,
		ylabel = L"B_H(t)", yguidefontsize = 14,
		tickfontsize = 12,
		title = "H = 0.5", titlefontsize = 16)
	p07 = Plots.plot(time_, h_07.ts, label = "", 
		lw = 3,
		xlabel = "Time", xguidefontsize = 14,
		ylabel = L"B_H(t)", yguidefontsize = 14,
		tickfontsize = 12,
		title = "H = 0.7", titlefontsize = 16)
	p09 = Plots.plot(time_, h_09.ts, label = "", 
		lw = 3,
		xlabel = "Time", xguidefontsize = 14,
		ylabel = L"B_H(t)", yguidefontsize = 14,
		tickfontsize = 12,
		title = "H = 0.9", titlefontsize = 16)
	all_together = Plots.plot(time_, all_tg.ts, label = ["H = 0.1" "H = 0.3" "H = 0.5" "H = 0.7" "H = 0.9"], 
		lw = 3,
		xlabel = "Time", xguidefontsize = 14,
		ylabel = L"B_H(t)", yguidefontsize = 14,
		tickfontsize = 12,
		title = "fBm comparison", titlefontsize = 16)
		
	Plots.plot(all_together, p01, p03, p05, p07, p09,
		dpi = 600, 
		layout=grid(2,3, widths=(4/12,4/12,4/12)), size=(1200,800), margin=5mm)
end

# ╔═╡ 0cc08a21-4853-49e1-aa9c-0fa9cd16721b
md"""
# Modeling Schema
As defined previously, we can define a SDE by using the same schema of a normal Brownian motion.

For example, for a SDE in the form:
```math
dX_t = b(X_t) dt + V(X_t) dW_t^H, X_0 = x_0 \in \mathbb{R} 
```
Where $dW_t^H$ is the fBm path.
Then, we have two schemas to solve this problem.
## First Order Euler Scheme
This is a similar scheme to a Bm, but with a fBm. Therefore, we have:
```math
\begin{align}
X_{t_{k+1}} &= X_{t_k} + b(X_{t_k})(t_{k+1} - t_k) + V(X_{t_k})(W_{t_{k+1}}^H - W_{t_k}^H)\\
&= X_{t_k} + b(X_{t_k})\delta t+V(X_{t_k})\delta W^H_{(t_k, t_{k+1})}

\end{align}
```
This scheme converges in 1/2<H
## Modified Euler Scheme
By expanding the Taylor serie to the second order we the serie converges in 1/3<H<1/2. This can be computed as:
```math
\begin{align}
X_{t_{k+1}} &= X_{t_k} + b(X_{t_k})\delta t+V(X_{t_k})\delta W^H_{(t_k, t_{k+1})} + \frac{1}{2} \frac{\partial V(X_{t_k})}{\partial x} V(X_{t_k})\delta t^2

\end{align}
```

## Simulations
In this case we are going to consider the Modified Euler Scheme.
Let's look at the same example as in Bm. 
```math
dX = \theta(\mu - X_t) dt + \sigma dW
```
By applying the Modified Euler Scheme we have:
```math
\begin{align}
b(X_{t_k}) &= \theta(\mu - X_t)\\
V(X_{t_k}) &= \sigma \\
\frac{\partial V(X_{t_k})}{\partial x} &= 0\\
\end{align}
```
Hence, we have that in this case, the Modified Euler Scheme is:
```math
\begin{align}
X_{t_{k+1}} &= X_{t_k} + b(X_{t_k})\delta t+V(X_{t_k})\delta W^H_{(t_k, t_{k+1})} + \frac{1}{2} \frac{\partial V(X_{t_k})}{\partial x} V(X_{t_k})\delta t^2\\
&= \theta(\mu - X_t) \delta t + \sigma \delta W^H_{(t_k, t_{k+1})}
\end{align}
```
---
Reference:

1. Wong, Y. C. C., & Bilokon, P. A. (2024). Simulation of Fractional Brownian Motion and Related Stochastic Processes in Practice: A Straightforward Approach.

"""

# ╔═╡ 7729196c-592e-45b9-bc8b-46858b9b5036
function fOPU(t0, dt, tf,x_0, H; theta = 0.7, mu = 1.5, sigma = 0.06)
	time = t0:dt:tf
	fbm_o = FBM(time, H)
	sample = rand(fbm_o, method=:chol)
	steps = length(time)
	v = zeros(steps,1)
	v[1] = x_0
	for i = 2:steps
		v[i] = v[i-1] + theta * (mu - v[i-1]) * dt + sigma * (sample[i] - sample[i-1])
	end
	return v
end

# ╔═╡ 38e847c8-d07b-4e1d-815d-639063894fff
function bmOUP(t0, dt, tf, x0; theta = 0.7, mu = 1.5, sigma = 0.06)
	time = t0:dt:tf
	steps = length(time)
	
	v = zeros(steps, 1)
	norm = Distributions.Normal(0,1)
	sample = rand(norm, steps)
	v[1] = x0
	for i = 2:steps
		v[i] = v[i-1]+theta*(mu - v[i-1]) * dt + sigma * sample[i] * sqrt(dt)
	end
	return v
end

# ╔═╡ 037296de-3076-4fa7-8e48-6a224d3190bc
begin
	t0_2 = 0
	dt_2 = 1/2^10
	tf_2 = 10
	time_2 = t0_2:dt_2:tf_2
	df_fplot = DataFrame(ts = Any[])
	for i = 1:10
		v = fOPU(t0_2, dt_2, tf_2, 0, 0.5)
		push!(df_fplot, [v])
	end
	df_plot = DataFrame(ts = Any[])
	for i  = 1:10
		v =  bmOUP(t0_2, dt_2, tf_2, 0)
		push!(df_plot, [v])
	end
	
	
	
	fopu_plot = Plots.plot(time_2, df_fplot.ts, label = "", dpi = 600,
		title = "fBm implementation, H = 0.5")
	opu_plot = Plots.plot(time_2, df_plot.ts, label = "", dpi = 600,
		title = "Bm implementation")
	Plots.plot(fopu_plot, opu_plot, 
	 layout=grid(1,2, widths=(1/2,1/2)), size=(800,400), margin=5mm)
end

# ╔═╡ 0e104630-56fc-4fdd-97b0-b3eadb81dc29


# ╔═╡ Cell order:
# ╟─1d0afe6c-2070-11ef-390b-87ebed423db6
# ╠═accbc1e0-0f92-4fba-8ac5-5bda66b67e21
# ╟─119c7c5e-aafd-464f-a585-a60995f157e9
# ╠═3d4c5a6a-d811-49b3-980e-5b42551261f9
# ╟─49cdd21e-8a7f-4fd8-ab09-6442f3ee4af8
# ╠═8b0ba491-3356-404c-a746-fe71a6d23e89
# ╟─0cc08a21-4853-49e1-aa9c-0fa9cd16721b
# ╠═7729196c-592e-45b9-bc8b-46858b9b5036
# ╠═38e847c8-d07b-4e1d-815d-639063894fff
# ╠═037296de-3076-4fa7-8e48-6a224d3190bc
# ╠═0e104630-56fc-4fdd-97b0-b3eadb81dc29
