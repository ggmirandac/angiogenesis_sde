import numpy as np

def brownian_motion(N, T, dim = 1):
    '''
    This function generates a Brownian Motion
    The function takes 3 arguments:
    N: number of steps
    T: time horizon
    dim: dimension of the Brownian Motion
    Output: 
    - a numpy array of shape (N, dim)
    The i-th row of the array is the value of the Brownian Motion at time i
    The j-th column is the array for the j-th dimension (x,y,z,...)
    '''
    dt = T/N
    dW = np.sqrt(dt)*np.random.randn(N, dim)
    W = np.cumsum(dW, axis = 1)
    return dW ,W

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 2000
    T = 3
    dW, W = brownian_motion(N, T, dim = 2)
    # trying winc
    for j in range(0,500):
        pass
    