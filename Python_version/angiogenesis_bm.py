from functions.brownian_motion import brownian_motion as bm
import functions.grad_div as gd
import functions.phi_calc as p
import parameters
import numpy as np
# definition of parameters is now in parameters.py

# initialization of arrays

nreps = int(1e0)

Xdata = dict() # this dictionary is going to store the data of each rep for the coordinate
Da_data = dict() # this dictionary is going to store the data of each rep for the diffusion gradient

for ix in range(nreps):
    Xem = np.zeros((parameters.N,2)) # array to store the coordinates
    Vem = np.zeros((parameters.N,2)) # array to store the velocity
    Xzero = np.array([0,0]) # initial condition coordinate
    Vzero = np.array([0,0]) # initial condition velocity
    Xem[0,:] = Xzero
    Vem[0,:] = Vzero
    # first angles
    theta = np.pi/2
    phi = np.pi/2
    theta_list = []
    Da_list = []

    # initialization of the loop for generate the motion of the sprout
    dW, W = bm(parameters.N, parameters.T, dim = 2)
    for j in range(0,parameters.L):
        print(j)
        