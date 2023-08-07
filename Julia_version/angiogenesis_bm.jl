include(joinpath("functions","BrowniaMotion.jl"))
using .BrownianMotion
nReps = 1e0

Xdata = Dict()
Da_data = Dict()

T = 3
N = 2000
dt = T/N

# parameters of the analysis
beta = 1/3 # h^-1
alpha = 40 * 1/3 # µm^2 h^-3
kappa = 10 # arbitrary parameter

for ix in 1:nReps

    

    # generate the fractional Brownian motion   

    bm = BrownianMotion.(N,T)

    Xzero = [0,0]
    Xtemp = Xzero

    

end
