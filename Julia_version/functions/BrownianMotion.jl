module BrownianMotion
    using DataFrames
    dt = T/N
    dW = sqrt(dt) * randn(2,N)
    W = cumsum(dW',dims = 2)
    dfW = DataFrame(bm_x = W[:,1], bm_y = W[:,2])
    export dfW
end


