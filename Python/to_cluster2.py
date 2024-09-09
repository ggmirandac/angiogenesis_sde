#%% Running
from angiosde import AngioSimulation
from os.path import join 
# parameters

n_reps = 100
Hurst_index = 0.75
n_steps = 3_000
dtau = .1
delta = 0.5  # TODO: review the delta effect over the simulation
mode = 'Simulate'

list_H = 
# for h in list_H:
for h in list_H:
    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                        xa=[0, 10_000],
                        mode="HitTime",
                        wall=1_000, )
    A_sim.simulate()
    A_sim.plot_sprouts()