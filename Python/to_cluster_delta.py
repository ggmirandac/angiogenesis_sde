#%% Running
from angiosde import AngioSimulation
import numpy as np
from os.path import join 

# parameters

n_reps = 100
Hurst_index = 0.5
n_steps = 3_000
dtau = .01
delta = 0.5  # TODO: review the delta effect over the simulation
mode = 'Simulate'

list_delta  = np.linspace(1/3, 3, 9)
# for h in list_H:
n_jobs = 2
for de in list_delta:

    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, de,
                        xa=[0, 10_000],
                        mode=mode,
                        wall=1_000)
    A_sim.simulate(n_jobs=n_jobs)
    A_sim.plot_sprouts()
    # folder = 'change_delta_075'
    # filen = 'hit_time_d_' + str(round(de, 3)).replace('.','_')    
    # A_sim.save_data(join(folder, filen))

# %%
