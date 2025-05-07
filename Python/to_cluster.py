#%% Running
# from angiosde import AngioSimulation, ConstantGradient, LinearGradient
from angiofsde import AngioSimulation, ConstantGradient, LinearGradient, ExponentialGradient
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
# parameters

n_reps = 1_000
# Hurst_index = 0.5
n_steps = 500_000
dtau = .001
delta = 1  # Done: review the delta effect over the simulation
mode = 'Simulate'

linear_gradient = LinearGradient(0.1, [0, 25], 25)
constant_gradient = ConstantGradient(0.1)
exponential_gradient = ExponentialGradient(0.1, [0, 25], 25)

list_H =  [0.5, 0.55, 0.60, 0.65, 0.7, 0.75]
gradients = [linear_gradient, constant_gradient, exponential_gradient]
# for h in list_H:
# TODO: Recoletar basura
for Grad in gradients:
    print(Grad.__class__.__name__)
    if Grad.__class__.__name__ == 'LinearGradient':
        name = 'linear'
    elif Grad.__class__.__name__ == 'ConstantGradient':
        name = 'constant'
    else:
        name = 'exponential'

    for h in list_H:
        A_sim = AngioSimulation(n_reps, h, n_steps, dtau, delta,
                            xa=[0, 25],
                            mode="HitTime",
                            wall=25,
                            only_ht=True,
                            Grad=Grad)
        A_sim.simulate(n_jobs=10)

        # A_sim.plot_sprouts()
        h_str = str(h).replace('.','_')
        file_name = join('personal_comp', f'{name}_hit_time_{h_str}')
        A_sim.save_hittimes(file_name)
        print(f'Hit time for Hurst index {h} saved')
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
