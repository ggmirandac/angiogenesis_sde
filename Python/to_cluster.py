#%% Running
from angiosde import AngioSimulation
from os.path import join 
# parameters

n_reps = 100
Hurst_index = 0.5
n_steps = 3_000
dtau = .1
delta = 0.5  # TODO: review the delta effect over the simulation
mode = 'Simulate'

list_H = [0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58,
          0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67,
          0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75]
# for h in list_H:
for h in list_H:
    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                        xa=[0, 10_000],
                        mode="HitTime",
                        wall=1_000, )
    A_sim.simulate()
    A_sim.plot_sprouts()
    # h_str = str(h).replace('.','_')
    # file_name = join('hit_time_d_05', f'hit_time{h_str}')
    # A_sim.save_data(file_name)

    # file_name = 'hit_time_' + str(h) 
    # A_sim.save_data(file_name)

# %%
