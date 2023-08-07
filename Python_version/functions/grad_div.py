def gradient_div(x,y,grad_value):
    '''
    This function generates the gradient function
    for a gradient that is devided by a variable y
    The function takes 3 arguments:
    x: the variable for the x coordinate
    y: the variable for the y coordinate
    grad_value: the value of the gradient
    Output:
    - the gradient value given the divition of the gradient
    '''

    if y == 0:
        grad = 0
    elif y < 0:
        grad  = -grad_value
    else:
        grad = grad_value
    return grad