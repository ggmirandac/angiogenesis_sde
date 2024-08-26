# =============================================================================
# Simulations on angiogenesis on fBm
# Date 15-8-2024
# =============================================================================

# %% Functions and classes for simulating angiogenesis
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from fbm.sim.davies_harte import DaviesHarteFBmGenerator
from fbm.sim.cholesky import CholeskyFBmGenerator
import time
import scipy.integrate as spi
import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from functools import partial

from tqdm import tqdm
import warnings


class Gradient:

    def __init__(self, a0, *args, **kwargs):
        self.initial_grad = a0

    def calculate_gradient(self, x):
        pass


class GradientConstant(Gradient):
    def __init__(self, a0):
        super().__init__(a0)

    def calculate_gradient(self, x):
        x_grad = 0
        y_grad = self.initial_grad / self.initial_grad

        return np.array([x_grad, y_grad])


class AngioSimulation:

    def __init__(self, n_reps, Hurst_index, n_steps, dtau, delta,
                 mode='Simulate',
                 Grad=GradientConstant(0.01),
                 xa=[0, 10],
                 wall=None,  # y coord of wall
                 ):
        self.n_reps = n_reps
        self.H = Hurst_index
        self.n_steps = n_steps
        self.dtau = dtau
        self.delta = delta
        self.mode = mode
        self.Gradient = Grad
        self.xa = np.array(xa)

        # storage_of_sprouts
        self.x_storage = {}
        self.v_storage = {}
        self.vd_storage = {}

        # storage of hiting_times
        self.hit_times = []
        self.wall = wall
        if mode == "HitTime":
            if wall is None:
                raise Exception("Need to define the wall to hit")

        # t intervals of plot
        closer_10_power = self.n_reps//10
        self.step = closer_10_power if closer_10_power > 0 else 1

    @staticmethod
    def sprout_generation(H, n_steps, dtau, delta, Gradient, xa):
        x_history = np.zeros((n_steps+1, 2))
        v_history = np.zeros((n_steps+1, 2))
        # v_resx, v_resy, v_randx, v_randy, v_chemx, v_chemy
        v_descriptions = np.zeros((n_steps + 1, 6))
        xi = np.array([0, 0])
        vi = np.array([0, 0])
        dW = np.zeros((n_steps, 2))
        dW[:, 0] = DaviesHarteFBmGenerator().generate_norm_fGn(
            H, size=n_steps) * dtau ** H
        dW[:, 1] = DaviesHarteFBmGenerator().generate_norm_fGn(
            H, size=n_steps) * dtau ** H

        theta = np.pi/2
        phi = AngioSimulation.phi_ang(xi, xa, theta)
        for step in range(0, n_steps):
            v_res = - vi * dtau  # resistance to movement
            v_rand = dW[step]
            v_chem = delta * Gradient.calculate_gradient(xi) * \
                np.sin(phi/2) * dtau

            # print(v_chem)
            v_descriptions[step + 1, :] = np.array(
                [v_res[0], v_res[1], v_rand[0], v_rand[1], v_chem[0], v_chem[1]])

            vi = vi + v_res + v_rand + v_chem
            xi = xi + vi * dtau
            x_history[step + 1, :] = xi
            v_history[step + 1, :] = vi
            xi_1 = x_history[step, :]
            theta = AngioSimulation.theta_ang(xi, xi_1)
            phi = AngioSimulation.phi_ang(xi, xa, theta)
            if theta is None or phi is None:
                crop_index = step
                x_history = x_history[:crop_index, :]
                v_history = v_history[:crop_index, :]
                v_descriptions = v_descriptions[:crop_index, :]
                return x_history, v_history, v_descriptions

        return x_history, v_history, v_descriptions
    @staticmethod
    def hit_generation(H, n_steps, dtau, delta, Gradient, xa, wall):

        x_history = np.zeros((n_steps + 1, 2))
        v_history = np.zeros((n_steps + 1, 2))
        v_descriptions = np.zeros((n_steps + 1, 6))

        xi = np.array([0, 0])
        vi = np.array([0, 0])
        dW = np.zeros((n_steps, 2))
        dW[:, 0] = DaviesHarteFBmGenerator().generate_fGn(
            H, size=n_steps) * dtau ** H
        dW[:, 1] = DaviesHarteFBmGenerator().generate_fGn(
            H, size=n_steps) * dtau ** H

        theta = np.pi/2
        phi = AngioSimulation.phi_ang(xi, xa, theta)
        for step in range(0, n_steps):
            v_res = - vi * dtau  # resistance to movement
            v_rand = dW[step]
            v_chem = delta * Gradient.calculate_gradient(xi) * np.sin(phi/2) * dtau
            v_descriptions[step + 1, :] = np.array(
                [v_res[0], v_res[1], v_rand[0], v_rand[1], v_chem[0], v_chem[1]])
            
            # print(v_chem)
            vi = vi + v_res + v_rand + v_chem
            xi = xi + vi * dtau
            x_history[step + 1, :] = xi
            v_history[step + 1, :] = vi
            xi_1 = x_history[step, :]
            theta = AngioSimulation.theta_ang(xi, xi_1)
            phi = AngioSimulation.phi_ang(xi, xa, theta)
            if phi is None or theta is None:
                crop_index = step
                x_history = x_history[:crop_index, :]
                v_history = v_history[:crop_index, :]
                v_descriptions = v_descriptions[:crop_index, :]
                return x_history, v_history, v_descriptions, step * dtau

            if xi[1] >= wall:
                crop_index = step
                x_history = x_history[:crop_index, :]
                v_history = v_history[:crop_index, :]
                v_descriptions = v_descriptions[:crop_index, :]
                return x_history, v_history, v_descriptions, step * dtau
            
        return x_history, v_history, v_descriptions , None    

    @staticmethod
    def theta_ang(xi, xi_1):
        x_axis = np.array([1, 0])
        x_dir = xi - xi_1
        x_dir_norm = np.linalg.norm(x_dir)
        num = np.dot(x_axis, x_dir)

        den = x_dir_norm


        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                theta = np.acos(
                    np.round(num/den, decimals=10)
                )

            except Warning as e:
                # print(num / den, num, den)
                theta = np.acos(
                    np.round(num/den, decimals=10)
                )
                
                return None
        return theta

    @staticmethod
    def phi_ang(xi, xa, theta):

        num = ((xa[0] - xi[0]) * np.cos(theta) +
               (xa[1] - xi[0]) * np.sin(theta))
        den = ((xa[0] - xi[0]) ** 2 + (xa[1] - xi[1]) ** 2) ** 1/2

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                phi = np.acos(
                    np.round(num/den, decimals=10)
                )

            except Warning as e:
                print(num / den, num, den)
                phi = np.acos(
                    np.round(num/den, decimals=3)
                )
                
                return None

        return phi

    def simulate(self, n_jobs):
        if self.mode == 'Simulation':
            init_time = time.time()

            results = Parallel(n_jobs=n_jobs)(delayed(AngioSimulation.sprout_generation)(
                self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa) for _ in range(self.n_reps))

            for i, result in enumerate(results):
                self.x_storage[f'ID - {i}'], self.v_storage[f'ID - {i}'], self.vd_storage[f'ID + {i}'] = result
                delta_time = (time.time() - init_time)
                minutes, seconds = divmod(delta_time, 60)

            print(
                f"Simulation of {self.n_reps} Sprouts generated. Time: {int(minutes)}:{seconds:.2f}")

        if self.mode == 'HitTime':
            init_time = time.time()

            for i in range(self.n_reps):
                result = AngioSimulation.hit_generation(
                    self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa,
                    self.wall
                )

                self.x_storage[f'ID - {i}'], self.v_storage[f'ID - {i}'], self.vd_storage[f'ID + {i}'], ht = result

                self.hit_times.append(ht)

                delta_time = (time.time() - init_time)
                minutes, seconds = divmod(delta_time, 60)
                if i % self.step == 0:
                    print(
                        f"Sprout {i+1} of {self.n_reps} Generated. Time: {int(minutes)}:{seconds:.2f}")

    def debbug(self):
        if self.mode == 'Simulation':
            init_time = time.time()
            for i in range(self.n_reps):
                result = AngioSimulation.sprout_generation(
                    self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa)
                self.x_storage[f'ID - {i}'], self.v_storage[f'ID - {i}'], self.vd_storage[f'ID + {i}'] = result
                delta_time = (time.time() - init_time)
                minutes, seconds = divmod(delta_time, 60)

            print(
                f"Simulation of {self.n_reps} Sprouts generated. Time: {int(minutes)}:{seconds:.2f}")

        if self.mode == 'HitTime':
            init_time = time.time()

            for i in range(self.n_reps):
                result = AngioSimulation.hit_generation(
                    self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa,
                    self.wall
                )

                self.x_storage[f'ID - {i}'], self.v_storage[f'ID - {i}'], self.vd_storage[f'ID + {i}'], ht = result
                self.hit_times.append(ht)
                delta_time = (time.time() - init_time)
                minutes, seconds = divmod(delta_time, 60)
                if i % self.step == 0:
                    print(
                        f"Sprout {i+1} of {self.n_reps} Generated. Time: {int(minutes)}:{seconds:.2f}")

    def plot_sprouts(self):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

        for sprout in self.x_storage.values():

            ax.plot(sprout[:, 0], sprout[:, 1])
            ax.scatter(sprout[-1, 0], sprout[-1, 1])
        plt.show()

    def plot_sprout_description(self):
        fig, ax = plt.subplots(5, 3, figsize=(20, 20))

        # plot_of_velocities
        # DONE: add All velocities # DONE
        for val in self.vd_storage.values():
            # unpack values
            v_resx = val[:, 0]
            v_resy = val[:, 1]
            v_randx = val[:, 2]
            v_randy = val[:, 3]
            v_chemx = val[:, 4]
            v_chemy = val[:, 5]

            time = np.arange(0, len(v_resx))
            # resistance to movement
            ax[0, 0].plot(time, v_resx)
            ax[0, 1].plot(time, v_resy)
            ax[0, 2].plot(v_resx, v_resy)

            # random velocity
            ax[1, 0].plot(time, v_randx)
            ax[1, 1].plot(time, v_randy)
            ax[1, 2].plot(v_randx, v_randy)

            # chemical attraction
            ax[2, 0].plot(time, v_chemx)
            ax[2, 1].plot(time, v_chemy)
            ax[2, 2].plot(time, v_chemy)

        # plot of velocities

        for val in self.v_storage.values():
            x_cord = val[:, 0]
            y_cord = val[:, 1]
            time = np.arange(0, len(x_cord))
            ax[3, 0].plot(time, x_cord)
            ax[3, 1].plot(time, y_cord)
            ax[3, 2].plot(x_cord, y_cord)

        # plot of sprout
        for val in self.x_storage.values():
            x_cord = val[:, 0]
            y_cord = val[:, 1]
            time = np.arange(0, len(x_cord))
            ax[4, 0].plot(time, x_cord)
            ax[4, 1].plot(time, y_cord)
            ax[4, 2].plot(x_cord, y_cord)
        # ploting specifications
        # V_RES
        ax[0, 0].set_title('Resistance Component - x')
        ax[0, 0].set_xlabel('Time')
        ax[0, 0].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[0, 0].set_ylabel('Velocity_x')

        ax[0, 1].set_title('Resistance Component - y')
        ax[0, 1].set_xlabel('Time')
        ax[0, 1].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[0, 1].set_ylabel('Velocity_y')

        ax[0, 2].set_title('Resistence Component - x,y')
        ax[0, 2].set_xlabel(r'Velocity_x')
        ax[0, 2].set_ylabel(r'Velocity_y')

        # V_RAND
        ax[1, 0].set_title('Random Component - x')
        ax[1, 0].set_xlabel('Time')
        ax[1, 0].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[1, 0].set_ylabel(r'Velocity_x')

        ax[1, 1].set_title('Random Component - y')
        ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[1, 1].set_ylabel(r'Velocity_y')

        ax[1, 2].set_title('Random Component - x,y')
        ax[1, 2].set_xlabel(r'Velocity_x')
        ax[1, 2].set_ylabel(r'Velocity_y')

        # V_CHEM
        ax[2, 0].set_title('Chemoattractant Component - x')
        ax[2, 0].set_xlabel('Time')
        ax[2, 0].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[2, 0].set_ylabel(r'Velocity_x')

        ax[2, 1].set_title('Chemoattractant Component - y')
        ax[2, 1].set_xlabel('Time')
        ax[2, 1].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[2, 1].set_ylabel(r'Velocity_y')

        ax[2, 2].set_title('Chemoattractant Component - x,y')
        ax[2, 2].set_xlabel(r'Velocity_x')
        ax[2, 2].set_ylabel(r'Velocity_y')

        # Velocities

        # Total Velocity
        ax[3, 0].set_title('Total Velocity')
        ax[3, 0].set_xlabel('Time')
        ax[3, 0].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[3, 0].set_ylabel(r'Velocity_x')

        ax[3, 1].set_title('Total Velocity')
        ax[3, 1].set_xlabel('Time')
        ax[3, 1].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[3, 1].set_ylabel(r'Velocity_y')

        ax[3, 2].set_title('Total Velocity')
        ax[3, 2].set_xlabel(r'Velocity_x')
        ax[3, 2].set_ylabel(r'Velocity_y')

        # Sprout

        ax[4, 0].set_title('Sprout - x')
        ax[4, 0].set_xlabel('Time')
        ax[4, 0].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[4, 0].set_ylabel(r'Position_x')

        ax[4, 1].set_title('Sprout - y')
        ax[4, 1].set_xlabel('Time')
        ax[4, 1].set_xticks(
            np.round(np.linspace(np.min(time), np.max(time), 10), 2))
        ax[4, 1].set_ylabel(r'Position_y')

        ax[4, 2].set_title('Sprout - x,y')
        ax[4, 2].set_xlabel(r'Position_x')
        ax[4, 2].set_ylabel(r'Position_y')
        fig.tight_layout()
        plt.show()

    def plot_autocorrelation(self, n_lags=100):

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # autocorrelation of sprouts
        vx_acf_array = np.zeros((n_lags+1, self.n_reps))
        vy_acf_array = np.zeros((n_lags+1, self.n_reps))
        for i, val in enumerate(self.v_storage.values()):
            vx_cord = val[:, 0]
            vy_cord = val[:, 1]
            vx_acf = sm.tsa.acf(vx_cord, nlags=n_lags)
            vy_acf = sm.tsa.acf(vy_cord, nlags=n_lags)
            vx_acf_array[:, i] = vx_acf
            vy_acf_array[:, i] = vy_acf

        mean_vx_acf = np.mean(vx_acf_array, axis=1)
        mean_vy_acf = np.mean(vy_acf_array, axis=1)
        se_vx_acf = stats.sem(vx_acf_array, axis=1)
        se_vy_acf = stats.sem(vy_acf_array, axis=1)

        ax[0, 0].plot(mean_vx_acf)
        ax[0, 0].fill_between(np.arange(n_lags+1), mean_vx_acf - se_vx_acf,
                              mean_vx_acf + se_vx_acf, alpha=0.5)

        ax[0, 1].plot(mean_vy_acf)
        ax[0, 1].fill_between(np.arange(n_lags+1), mean_vy_acf - se_vy_acf,
                              mean_vy_acf + se_vy_acf, alpha=0.5)

        # sprouts
        x_acf_array = np.zeros((n_lags+1, self.n_reps))
        y_acf_array = np.zeros((n_lags+1, self.n_reps))
        for i, val in enumerate(self.x_storage.values()):
            x_cord = val[:, 0]
            y_cord = val[:, 1]
            x_acf = sm.tsa.acf(x_cord, nlags=n_lags)
            y_acf = sm.tsa.acf(y_cord, nlags=n_lags)
            x_acf_array[:, i] = x_acf
            y_acf_array[:, i] = y_acf

        mean_x_acf = np.mean(x_acf_array, axis=1)
        mean_y_acf = np.mean(y_acf_array, axis=1)
        se_x_acf = stats.sem(x_acf_array, axis=1)
        se_y_acf = stats.sem(y_acf_array, axis=1)

        ax[1, 0].plot(mean_x_acf)
        ax[1, 0].fill_between(np.arange(n_lags+1), mean_x_acf - se_x_acf,
                              mean_x_acf + se_x_acf, alpha=0.5)

        ax[1, 1].plot(mean_y_acf)
        ax[1, 1].fill_between(np.arange(n_lags+1), mean_y_acf - se_y_acf,
                              mean_y_acf + se_y_acf, alpha=0.5)

        # ploting specifications

        # Sprout
        ax[0, 0].set_title('Autocorrelation of Velocities - x')
        ax[0, 0].set_xlabel('Lag')
        ax[0, 0].set_ylabel('ACF')

        ax[0, 1].set_title('Autocorrelation of Velocities - y')
        ax[0, 1].set_xlabel('Lag')
        ax[0, 1].set_ylabel('ACF')

        # Velocities

        ax[1, 0].set_title('Autocorrelation of Sprout - x')
        ax[1, 0].set_xlabel('Lag')
        ax[1, 0].set_ylabel('ACF')

        ax[1, 1].set_title('Autocorrelation of Sprout - y')
        ax[1, 1].set_xlabel('Lag')
        ax[1, 1].set_ylabel('ACF')

        fig.suptitle(f'Autocorrelation analysis - H = {self.H}')
        fig.tight_layout()

        plt.show()

    def plot_hit(self):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        if len(self.hit_times) == 0:
            raise Exception("Need to simulate the hit times of the Sprouts")

        non_reach = self.hit_times.count(None)
        porcentage_nr = np.round(non_reach/len(self.hit_times) * 100, 2)

        reaching = [ht for ht in self.hit_times if ht is not None]
        vp = ax.violinplot(reaching, [1])
        bp = ax.boxplot(reaching, [1])
        ax.legend(
            [f'n° non-reaching sprouts: \n - {non_reach}\n - {porcentage_nr}%'])
        ax.set_xticks([1], [f'H = {self.H}'])

        ax.set_yticks(
            np.round(np.linspace(np.min(reaching),
                     np.max(reaching), 10), 1),

        )

        plt.show()

    def save_data(self):
        hitting_times = self.hit_times
        sprouts = self.x_storage
        velocities = self.v_storage
        velocities_description = self.vd_storage
         
        # create pandas dataframe from the data
        hit_pd = pd.DataFrame(hitting_times, columns=['Hitting Time'])
        # sprouts_pd = pd.DataFrame(sprouts, columns=['Sprouts_x', 'Sprouts_y'])
        # velocities_pd = pd.DataFrame(velocities, columns=['Velocities_x', 'Velocities_y'])
        hit_pd.to_csv('hit_times.csv', index=False)
        

# %% Main body
if __name__ == "__main__":
    n_reps = 100
    Hurst_index = 0.99
    n_steps = 10_000
    dtau = 1
    delta = 3  # TODO: review the delta effect over the simulation
    mode = 'HitTime'
    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                            xa=[0, 10_000],
                            mode=mode,
                            wall=1_000, )
    A_sim.simulate(n_jobs=1)
    if mode == 'Simulation':

        # A_sim.plot_sprout_description()
        A_sim.plot_autocorrelation()

    elif mode == 'HitTime':
        A_sim.plot_hit()
        A_sim.plot_sprouts()
        A_sim.plot_sprout_description()
        A_sim.save_data()
# %%
