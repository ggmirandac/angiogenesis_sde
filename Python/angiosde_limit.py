# =============================================================================
# Simulations on angiogenesis on fBm
# Date 15-8-2024
# =============================================================================

# %% Functions and classes for simulating angiogenesis
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap handling
import matplotlib.colors as mcolors  # For normalization
from fbm.sim.davies_harte import DaviesHarteFBmGenerator
from fbm.sim.cholesky import CholeskyFBmGenerator
import time
from matplotlib.collections import LineCollection
# import scipy.integrate as spi
import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats
from scipy.spatial import distance
from joblib import Parallel, delayed
from functools import partial

from tqdm import tqdm
import warnings


class Gradient:

    def __init__(self, a0, *args, **kwargs):
        self.initial_grad = a0

    def calculate_gradient(self, x):
        pass


class ConstantGradient(Gradient):
    def __init__(self, a0):
        super().__init__(a0)

    def calculate_gradient(self, x):
        x_grad = 0
        y_grad = self.initial_grad / self.initial_grad

        return np.array([x_grad, y_grad])

class LinearGradient(Gradient):
    def __init__(self, a0, xa, wall ):
        '''
        a0: Concentration at source
        
        xa: x coordinate of the source
        
        min_gradient: minimum gradient
        '''
        super().__init__(a0)
        self.xa = xa
        
        self.wall = wall 
    def calculate_gradient(self, x):
        x_grad = 0
        y_grad = x[1] * self.initial_grad / self.wall
        
        return np.array([x_grad, y_grad]) / self.initial_grad
        
class ExponentialGradient(Gradient):
    def __init__(self, a0, xa, wall):
        super().__init__(a0)
        self.xa = xa
        self.wall = wall
    def calculate_gradient(self, x):
        x_grad = 0
        A = self.initial_grad
        B = np.log(1-self.initial_grad/A*0.9)/(-self.wall/2)
        y_grad = A * (1 - np.exp(-x[1]*B))/self.initial_grad
        return np.array([x_grad, y_grad])

        
        
        
class AngioSimulation:

    def __init__(self, n_reps, Hurst_index, n_steps, dtau, delta,
                 mode='Simulate',
                 Grad=ConstantGradient(0.01),
                 only_ht = True,
                 xa = [0, 10],
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
        self.only_ht = only_ht # true -> store all the history, else only hitting time
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
    def sprout_generation(H, n_steps, dtau, delta, Gradient, xa, wall,
                       only_ht):
        x_history = np.zeros((n_steps + 1, 2))
        v_history = np.zeros((n_steps + 1, 2))
        v_descriptions = np.zeros((n_steps + 1, 6))

        xi = np.array([0, 0])
        vi = np.array([0, 0])
        dW = np.zeros((n_steps, 2))
        # by the definition in the documentation if T = 0 and size = size, the interval of time is 
        # defined as 1, therefore is consistent with the definition of multiplying by dtau ** H
        dW[:, 0] = DaviesHarteFBmGenerator().generate_fGn(
            H, size=n_steps) * dtau ** H
        dW[:, 1] = DaviesHarteFBmGenerator().generate_fGn(
            H, size=n_steps) * dtau ** H

        theta = np.pi/2
        phi = AngioSimulation.phi_ang(xi, xa, theta)
        for step in range(0, n_steps):
            v_res = - vi * dtau  # resistance to movement
            v_rand = dW[step]
            v_chem = delta * Gradient.calculate_gradient(xi) * np.sin(np.abs(phi/2)) * dtau
            v_descriptions[step + 1, :] = np.array(
                [v_res[0], v_res[1], v_rand[0], v_rand[1], v_chem[0], v_chem[1]])
            
            # print(v_chem)
            
            # define
            # last step
            xi_1 = np.copy(xi)
            
            xi = xi + vi * dtau
            hit_lower_wall = False
            if xi[1] < 0:
                xi = np.array([xi[0], 0])
                # now we have to redifine in the last step to apply to the 
                # calculations for the next step
                # we have to 
                # definition of velocity in the step step-1
                vi = (xi - xi_1) / dtau 
            
            vi = vi + v_res + v_rand + v_chem
            x_history[step + 1, :] = xi
            v_history[step + 1, :] = vi
            xi_1 = x_history[step, :]
            theta = AngioSimulation.theta_ang(xi, xi_1)
            xa = np.array([xi[0], wall])
            phi = AngioSimulation.phi_ang(xi, xa, theta)
            
           
            if xi[1] >= wall:
                crop_index = step+1
                x_history = x_history[:crop_index, :]
                v_history = v_history[:crop_index, :]
                v_descriptions = v_descriptions[:crop_index, :]
                # if hit the wall and only ht return the hit time
                if only_ht:
                    return step*dtau
                # if hit the wall and not only ht return the history and time hit
                return x_history, v_history, v_descriptions, step * dtau
        # if only ht and not hit the call return None

        if only_ht:
            return None
        else:
            return x_history, v_history, v_descriptions , None    

    @staticmethod
    def theta_ang(xi, xi_1):

        x_trans = xi[0] - xi_1[0]
        y_trans = xi[1] - xi_1[1]
        theta = np.arctan2(y_trans, x_trans)

        return theta

    @staticmethod
    def phi_ang(xi, xa, theta):

        num =((xa[0] - xi[0]) * np.cos(theta)) + ((xa[1] - xi[1]) * np.sin(theta))
        den = np.sqrt((xa[0] - xi[0]) ** 2 + (xa[1] - xi[1]) ** 2)
    
    
        return np.arccos(num/den)

    def simulate(self, n_jobs = 1):
        # modes
        # Simulate: generate sprouts
        # HitTime: generate hit times
        # SimulateAndHit: generates both
        
        if self.mode == 'Simulate':
            # init_time = time.time()
            results = Parallel(n_jobs=n_jobs)(delayed(AngioSimulation.sprout_generation)(
                self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa, self.wall, False) for _ in range(self.n_reps))

            for i, result in enumerate(results):
                self.x_storage[f'ID - {i}'], self.v_storage[f'ID - {i}'], self.vd_storage[f'ID + {i}'], _ = result
                
                # delta_time = (time.time() - init_time)
                # minutes, seconds = divmod(delta_time, 60)

            # print(
            #     f"Simulation of {self.n_reps} Sprouts generated. Time: {int(minutes)}:{seconds:.2f}")

        if self.mode == 'HitTime':
            results = Parallel(n_jobs = n_jobs, backend='loky', verbose=1)(delayed(AngioSimulation.sprout_generation)(
                self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa, self.wall, True
            ) for _ in range(self.n_reps))
            
            self.hit_times = results
        
        if self.mode == 'SimulateAndHit':


            results = Parallel(n_jobs=n_jobs)(delayed(AngioSimulation.sprout_generation)(
                self.H, self.n_steps, self.dtau, self.delta, self.Gradient, self.xa, self.wall, False) for _ in range(self.n_reps))
            
            for i, result in enumerate(results):

                self.x_storage[f'ID - {i}'], self.v_storage[f'ID - {i}'], self.vd_storage[f'ID + {i}'], ht = result

                self.hit_times.append(ht)


    def plot_sprouts(self, title, show = True):
        if self.Gradient.__class__.__name__ == 'ConstantGradient':
            
            fig, ax = plt.subplots(figsize=(8,8), dpi=600)
            
            # ax[0].set_size_inches(2, 10)
            # Initialize min/max variables
            minx, maxx = np.inf, -np.inf
            miny, maxy = 0, self.wall
            
            ########### Plot each sprout ###########
            for sprout in self.x_storage.values():
                minx = min(minx, np.min(sprout[:, 0]))
                maxx = max(maxx, np.max(sprout[:, 0]))
                ax.plot(sprout[:, 0], sprout[:, 1], zorder=3, linewidth=2)
                ax.scatter(sprout[-1, 0], sprout[-1, 1])
            
            # # Create meshgrid for gradient
            # X_coords = np.linspace(minx, maxx, 10)
            # Y_coords = np.linspace(miny, maxy, 10)
            # X, Y = np.meshgrid(X_coords, Y_coords)
            
            # # Calculate gradient vectors
            # X_grad, Y_grad = np.zeros_like(X), np.zeros_like(Y)
            # for i in range(10):
            #     for j in range(10):
            #         X_grad[i, j], Y_grad[i, j] = self.Gradient.calculate_gradient([X[i, j], Y[i, j]])
            
            # # Calculate magnitude and normalize
            
            # if self.Gradient.__class__.__name__ == 'ConstantGradient':
            #     color_mag = np.ones_like(X_grad)
            #     norm = mcolors.Normalize(vmin=0, vmax=1)
            # else:
            #     color_mag = np.sqrt(X_grad**2 + Y_grad**2)
            #     norm = mcolors.Normalize(vmin=np.min(color_mag), vmax=np.max(color_mag))
            # colormap = cm.plasma  # Choose the colormap
            
            # # Flatten arrays for quiver
            # X_flat, Y_flat = X.ravel(), Y.ravel()
            # U_flat, V_flat = X_grad.ravel(), Y_grad.ravel()
            # color_mag_flat = color_mag.ravel()
            
            # # Map colors
            # colors = colormap(norm(color_mag_flat))
            
            # # Plot gradient vectors using quiver
            # ax[1].quiver(X_flat, Y_flat, U_flat, V_flat, color=colors, 
            #         zorder=1, alpha=0.7)
            
            # # Add colorbar
            # sm = cm.ScalarMappable(cmap=colormap, norm=norm)  # Create a ScalarMappable
            # sm.set_array([])  # Empty array to link with the colorbar
            # cbar = fig.colorbar(sm, ax=ax, orientation='vertical')  # Add colorbar
            # cbar.set_label('Gradient Magnitude [a.u.]', fontsize = 18)  # Label the colorbar
            # # cbar.set_ticks(cbar.get_ticks())
            # cbar.set_ticklabels(np.round(cbar.get_ticks(),1), fontsize = 15)  # Update the tick labels
        
            # Set plot limits
            ax.set_xlabel('X [a.u.]', fontsize=15)
            # ax[1].set_ylabel('Y [a.u.]', fontsize=15)
            
            ax.hlines(0, minx, maxx, color='black', linestyle='--', linewidth=3)
            ax.hlines(self.wall, minx, maxx, color='black', linestyle='--', linewidth=3)
            ax.set_yticks([np.round(x, 1) for x in np.linspace(0, self.wall, 5)])
            ax.set_yticklabels(ax.get_yticks(), fontsize=12)
            xticks = ax.get_xticks()
            ax.set_xticks(xticks)
            ax.set_xticklabels(ax.get_xticks(), fontsize=12)
            
            ax.set_xlim([minx-0.1, maxx+0.1])
            ax.set_ylim([miny-0.1, maxy+0.1])
            ax.set_title(title, fontsize=15)  
            # Now we plot the gradient in the first plot
            
            
        else: 
            fig, ax = plt.subplots(1, 2 ,figsize=(9,8), dpi=600,
                                gridspec_kw={'width_ratios': [1, 5], 'wspace': 0.2}, sharey=True)
            
            # ax[0].set_size_inches(2, 10)
            # Initialize min/max variables
            minx, maxx = np.inf, -np.inf
            miny, maxy = 0, self.wall
            
            ########### Plot each sprout ###########
            for sprout in self.x_storage.values():
                minx = min(minx, np.min(sprout[:, 0]))
                maxx = max(maxx, np.max(sprout[:, 0]))
                ax[1].plot(sprout[:, 0], sprout[:, 1], zorder=3, linewidth=2)
                ax[1].scatter(sprout[-1, 0], sprout[-1, 1])
            
            # # Create meshgrid for gradient
            # X_coords = np.linspace(minx, maxx, 10)
            # Y_coords = np.linspace(miny, maxy, 10)
            # X, Y = np.meshgrid(X_coords, Y_coords)
            
            # # Calculate gradient vectors
            # X_grad, Y_grad = np.zeros_like(X), np.zeros_like(Y)
            # for i in range(10):
            #     for j in range(10):
            #         X_grad[i, j], Y_grad[i, j] = self.Gradient.calculate_gradient([X[i, j], Y[i, j]])
            
            # # Calculate magnitude and normalize
            
            # if self.Gradient.__class__.__name__ == 'ConstantGradient':
            #     color_mag = np.ones_like(X_grad)
            #     norm = mcolors.Normalize(vmin=0, vmax=1)
            # else:
            #     color_mag = np.sqrt(X_grad**2 + Y_grad**2)
            #     norm = mcolors.Normalize(vmin=np.min(color_mag), vmax=np.max(color_mag))
            # colormap = cm.plasma  # Choose the colormap
            
            # # Flatten arrays for quiver
            # X_flat, Y_flat = X.ravel(), Y.ravel()
            # U_flat, V_flat = X_grad.ravel(), Y_grad.ravel()
            # color_mag_flat = color_mag.ravel()
            
            # # Map colors
            # colors = colormap(norm(color_mag_flat))
            
            # # Plot gradient vectors using quiver
            # ax[1].quiver(X_flat, Y_flat, U_flat, V_flat, color=colors, 
            #         zorder=1, alpha=0.7)
            
            # # Add colorbar
            # sm = cm.ScalarMappable(cmap=colormap, norm=norm)  # Create a ScalarMappable
            # sm.set_array([])  # Empty array to link with the colorbar
            # cbar = fig.colorbar(sm, ax=ax, orientation='vertical')  # Add colorbar
            # cbar.set_label('Gradient Magnitude [a.u.]', fontsize = 18)  # Label the colorbar
            # # cbar.set_ticks(cbar.get_ticks())
            # cbar.set_ticklabels(np.round(cbar.get_ticks(),1), fontsize = 15)  # Update the tick labels
        
            # Set plot limits
            ax[1].set_xlabel('X [a.u.]', fontsize=15)
            # ax[1].set_ylabel('Y [a.u.]', fontsize=15)
            
            ax[1].hlines(0, minx, maxx, color='black', linestyle='--', linewidth=3)
            ax[1].hlines(self.wall, minx, maxx, color='black', linestyle='--', linewidth=3)
            ax[1].set_yticks([np.round(x, 1) for x in np.linspace(0, self.wall, 5)])
            xticks = ax[1].get_xticks()
            ax[1].set_xticks(xticks)
            ax[1].set_xticklabels(ax[1].get_xticks(), fontsize=12)
            
            ax[1].set_xlim([minx-0.1, maxx+0.1])
            ax[1].set_ylim([miny-0.1, maxy+0.1])
            ax[1].set_title(title, fontsize=15)  
            # Now we plot the gradient in the first plot
            
            ############ Plot gradient ###########
            
            grad_y = np.linspace(0, self.wall, 100)
            
            # Calculate the gradient values at each point
            funct = np.array([self.Gradient.calculate_gradient([0, y]) for y in grad_y])
            
            # Extract the x-component (magnitude to be visualized)
            colors_grad = funct[:, 1]  
            
            # Normalize the gradient values for the colormap
            norm = mcolors.Normalize(vmin=0, vmax=1)
            
            
            # Create line segments for color mapping
            points = np.array([colors_grad, grad_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a LineCollection and set colors based on the magnitude
            lc = LineCollection(segments, colors = 'tab:purple',norm=norm, linewidth=2)
            lc.set_array(colors_grad)  # Map colors to the x-component magnitude
            
            # Add the LineCollection to the plot
            ax[0].add_collection(lc)
            
            # Adjust plot limits and labels
            # if consttant make it 
    
            ax[0].set_xlim([np.min(colors_grad) - 0.01, np.max(colors_grad) + 0.1])
            ax[0].set_xticks([0, 0.5, 1])
            ax[0].set_xticklabels([0, 0.5, 1], fontsize=15)
            ax[0].set_ylim([0, self.wall])
            ax[0].set_xlabel('Magnitude [a.u.]', fontsize = 15)
            ax[0].set_ylabel('Y [a.u.]', fontsize = 15)
            ax[0].invert_xaxis()  # Invert x-axis if needed
            ax[0].set_title('Gradient', fontsize = 15)
            ax[0].set_yticklabels(ax[0].get_yticks(), fontsize=12)
        if show:
            plt.show()  # Display the plot
        # fig.tight_layout()
        return fig, ax

    def plot_sprout_description(self):
        fig, ax = plt.subplots(5, 3, figsize=(20, 20))

        # plot_of_velocities
        # DONE: add All velocities # DONE
        max_time = 0
        min_time = np.inf   
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

            if max(time) > max_time:
                max_time = max(time)
            if min(time) < min_time:
                min_time = min(time)
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
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[0, 0].set_ylabel('Velocity_x')

        ax[0, 1].set_title('Resistance Component - y')
        ax[0, 1].set_xlabel('Time')
        ax[0, 1].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[0, 1].set_ylabel('Velocity_y')

        ax[0, 2].set_title('Resistence Component - x,y')
        ax[0, 2].set_xlabel(r'Velocity_x')
        ax[0, 2].set_ylabel(r'Velocity_y')

        # V_RAND
        ax[1, 0].set_title('Random Component - x')
        ax[1, 0].set_xlabel('Time')
        ax[1, 0].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[1, 0].set_ylabel(r'Velocity_x')

        ax[1, 1].set_title('Random Component - y')
        ax[1, 1].set_xlabel('Time')
        ax[1, 1].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[1, 1].set_ylabel(r'Velocity_y')

        ax[1, 2].set_title('Random Component - x,y')
        ax[1, 2].set_xlabel(r'Velocity_x')
        ax[1, 2].set_ylabel(r'Velocity_y')

        # V_CHEM
        ax[2, 0].set_title('Chemoattractant Component - x')
        ax[2, 0].set_xlabel('Time')
        ax[2, 0].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[2, 0].set_ylabel(r'Velocity_x')

        ax[2, 1].set_title('Chemoattractant Component - y')
        ax[2, 1].set_xlabel('Time')
        ax[2, 1].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[2, 1].set_ylabel(r'Velocity_y')

        ax[2, 2].set_title('Chemoattractant Component - x,y')
        ax[2, 2].set_xlabel(r'Velocity_x')
        ax[2, 2].set_ylabel(r'Velocity_y')

        # Velocities

        # Total Velocity
        ax[3, 0].set_title('Total Velocity')
        ax[3, 0].set_xlabel('Time')
        ax[3, 0].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[3, 0].set_ylabel(r'Velocity_x')

        ax[3, 1].set_title('Total Velocity')
        ax[3, 1].set_xlabel('Time')
        ax[3, 1].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[3, 1].set_ylabel(r'Velocity_y')

        ax[3, 2].set_title('Total Velocity')
        ax[3, 2].set_xlabel(r'Velocity_x')
        ax[3, 2].set_ylabel(r'Velocity_y')

        # Sprout

        ax[4, 0].set_title('Sprout - x')
        ax[4, 0].set_xlabel('Time')
        ax[4, 0].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
        ax[4, 0].set_ylabel(r'Position_x')

        ax[4, 1].set_title('Sprout - y')
        ax[4, 1].set_xlabel('Time')
        ax[4, 1].set_xticks(
            np.round(np.linspace(min_time, max_time, 10), 2))
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

    def save_hittimes(self, file_name):
        hitting_times = self.hit_times

         
        # create pandas dataframe from the data
        hit_pd = pd.DataFrame(hitting_times, columns=['Hitting Time'])
        # sprouts_pd = pd.DataFrame(sprouts, columns=['Sprouts_x', 'Sprouts_y'])
        # velocities_pd = pd.DataFrame(velocities, columns=['Velocities_x', 'Velocities_y'])
        hit_pd.to_csv(f'{file_name}.csv', index=False)
        
\
# %% Main body
if __name__ == "__main__":
    n_reps = 10
    Hurst_index = 0.95
    n_steps = 30_000
    dtau = 1e-3
    delta = 0.95  # Done: review the delta effect over the simulation
    mode = 'Simulate'
    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                            xa=[0, 10_000],
                            mode=mode,
                            wall=50, )

    # Note
        # The more hurst index increases the less the delta imports
        # Done: Analyze the distributions of the same hurst index at different delta coefficient
    A_sim.simulate(n_jobs=1)
    if mode == 'Simulate':
        A_sim.plot_sprouts()
        # A_sim.plot_sprout_description()
        #A_sim.plot_autocorrelation()

    elif mode == 'HitTime':
        A_sim.plot_hit()
        A_sim.plot_sprouts()
        A_sim.plot_sprout_description()
        A_sim.save_data()
# %%
