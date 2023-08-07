# Path: funtions/gradient_div.jl

function gradient_div(x,y,grad_value)
    if y == 0
        return 0
    elseif y < 0
        return -grad_value
    elseif y > 0
        return grad_value
    end
end 


