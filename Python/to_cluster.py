#%% Running
from angiosde import AngioSimulation
from os.path import join 
import numpy as np
import matplotlib.pyplot as plt
# parameters

n_reps = 1000
Hurst_index = 0.5
n_steps = 100_000
dtau = .001
delta = 1  # TODO: review the delta effect over the simulation
mode = 'Simulate'


list_H =  [0.5]
# for h in list_H:
for h in list_H:
    A_sim = AngioSimulation(n_reps, h, n_steps, dtau, delta,
                        xa=[0, 10_000],
                        mode="HitTime",
                        wall=25, 
                        only_ht=True)
    A_sim.simulate(n_jobs=10)
    hit_times = A_sim.hit_times
    
    z_o_ht = [1 if x == None else 0 for x in hit_times]
    print(sum(z_o_ht))
    plt.hist([np.nan if x == None else x for x in hit_times])
    
    # h_str = str(h).replace('.','_')
    # file_name = join('hit_time_d_05', f'hit_time{h_str}')
    # A_sim.save_data(file_name)

    # file_name = 'hit_time_' + str(h) 
    # A_sim.save_data(file_name)



# %%

# %%
