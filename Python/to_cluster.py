#%% Running
from angiosde import AngioSimulation

# parameters

n_reps = 100
Hurst_index = 0.5
n_steps = 10_000
dtau = 1
delta = 3  # TODO: review the delta effect over the simulation
mode = 'HitTime'
A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                        xa=[0, 10_000],
                        mode=mode,
                        wall=1_000, )

A_sim.simulate(n_jobs = 1)
A_sim.save_data()

# %%
