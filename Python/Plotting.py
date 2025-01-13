#%% Running
# from angiosde import AngioSimulation, ConstantGradient, LinearGradient
from angiosde_limit import AngioSimulation, ConstantGradient, LinearGradient, ExponentialGradient
from os.path import join 
import numpy as np
import matplotlib.pyplot as plt
# parameters

n_reps = 5
# Hurst_index = 0.5
n_steps = 500_000
dtau = .001
delta = 1  # Done: review the delta effect over the simulation
mode = 'Simulate'
wall = 50
linear_gradient = LinearGradient(0.1, [0, wall], wall)
constant_gradient = ConstantGradient(0.1)
exponential_gradient = ExponentialGradient(0.1, [0, wall], wall)

list_H =  [0.85]
# for h in list_H:
# TODO: Recoletar basura

for h in list_H:
    A_sim = AngioSimulation(n_reps, h, n_steps, dtau, delta,
                        xa=[0, wall],
                        mode="Simulate",
                        wall=wall, 
                        only_ht=True, 
                        Grad=constant_gradient)
    A_sim.simulate(n_jobs=10)
    A_sim.plot_sprouts('Sprout path H =' + str(h)+' Constant Gradient')
    # A_sim.save_data(file_na
# %%
