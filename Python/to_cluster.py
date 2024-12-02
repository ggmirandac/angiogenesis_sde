#%% Running
# from angiosde import AngioSimulation, ConstantGradient, LinearGradient
from angiosde_limit import AngioSimulation, ConstantGradient, LinearGradient, ExponentialGradient
from os.path import join 
import numpy as np
import matplotlib.pyplot as plt
# parameters

n_reps = 10
# Hurst_index = 0.5
n_steps = 500_000
dtau = .001
delta = 1  # Done: review the delta effect over the simulation
mode = 'Simulate'

# lin_gradient = LinearGradient(0.1, [0, 25], 25)
# const_gradient = Linea(0.1)

list_H =  [0.75]
# for h in list_H:
for h in list_H:
    A_sim = AngioSimulation(n_reps, h, n_steps, dtau, delta,
                        xa=[0, 25],
                        mode="Simulate",
                        wall=25, 
                        only_ht=True, 
                        Grad=ExponentialGradient(0.1, [0, 25], 25))
    A_sim.simulate(n_jobs=10)

    A_sim.plot_sprouts()
    # h_str = str(h).replace('.','_')
    # file_name = join('hit_time_d_05', f'hit_time{h_str}')
    # A_sim.save_data(file_name)

    # file_name = 'hit_time_' + str(h) 
    # A_sim.save_data(file_name)

#%%

# mesh_grid = np.meshgrid(np.linspace(-10, 10, 5), np.linspace(-10, 10, 5), indexing='ij')
# X, Y = mesh_grid
# Grad = GradientConstant(0.1)
# X_direction = np.zeros_like(X)
# Y_direction = np.zeros_like(Y)
# for i in range(5):
#     for j in range(5):
#         X_direction[i, j], Y_direction[i, j] = Grad.calculate_gradient([X[i, j], Y[i, j]])



# plt.quiver(X, Y, X_direction, Y_direction)

# %%

# %%
