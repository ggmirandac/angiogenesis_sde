#%% Running
from angiosde import AngioSimulation
from os.path import join 
import numpy as np
# parameters

n_reps = 100
Hurst_index = 0.5
n_steps = 10_000
dtau = .01
delta = 1  # TODO: review the delta effect over the simulation
mode = 'Simulate'

list_H =  np.linspace(0.5, 0.6, 10)
# for h in list_H:
for h in list_H:
    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                        xa=[0, 10_000],
                        mode="HitTime",
                        wall=200, )
    A_sim.simulate()
    A_sim.plot_sprouts()
    # h_str = str(h).replace('.','_')
    # file_name = join('hit_time_d_05', f'hit_time{h_str}')
    # A_sim.save_data(file_name)

    # file_name = 'hit_time_' + str(h) 
    # A_sim.save_data(file_name)


ah# %%
