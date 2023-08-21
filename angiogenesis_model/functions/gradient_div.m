function gradient = gradient_div(x,y,grad_value)
    if y == 0
        grad = 0;
    elseif y < 0
        grad = -grad_value;
    elseif y > 0
        grad = grad_value;
    end
    gradient = [0; grad];
end
