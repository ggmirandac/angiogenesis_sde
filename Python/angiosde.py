# =============================================================================
# Simulations on angiogenesis on fBm
# Date 15-8-2024
# =============================================================================

#%% Functions and classes for simulating angiogenesis
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from fbm.sim.davies_harte import DaviesHarteFBmGenerator
from fbm.sim.cholesky import CholeskyFBmGenerator
import time

class Gradient:
    def __init__(self, a0, *args,**kwargs):
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
                 mode = 'Simulate', 
                 Grad = GradientConstant(0.01),
                 xa = [0,10]):
        self.n_reps = n_reps
        self.H = Hurst_index
        self.n_steps = n_steps  
        self.dtau = dtau
        self.delta = delta
        self.mode = mode
        self.Gradient = Grad
        self.method = 'cholesky'
        self.xa = np.array(xa)
        
        # storage_of_sprouts
        self.x_storage = {}
        self.v_storage = {}
    def simulate(self):
        if self.mode == 'Simulate':
            
            init_time = time.time()
            for i in range(self.n_reps): 
                self.x_storage[f'ID - {i}'], self.v_storage[f'ID + {i}']= self.sprout_generation()
                delta_time = (time.time() - init_time)
                minutes, seconds = divmod(delta_time, 60)
                print(f"Sprout {i+1} of {self.n_reps} Generated. Time: {int(minutes)}:{seconds:.2f}")
            
        if self.mode == 'HittingTime':
            self.hitting_time()
    
    def sprout_generation(self):    
        x_history = np.zeros((self.n_steps+1, 2))
        v_history = np.zeros((self.n_steps+1, 2))
        xi = np.array([0,0])
        vi = np.array([0,0])
        dW = np.zeros((self.n_steps, 2))
        dW[:, 0] = DaviesHarteFBmGenerator().generate_fGn(self.H, size = self.n_steps) * self.dtau ** self.H
        dW[:, 1] = DaviesHarteFBmGenerator().generate_fGn(self.H, size = self.n_steps) * self.dtau ** self.H
        
        
            
        theta = np.pi/2
        phi = self.phi_ang(xi, self.xa, theta)
        for step in range(0, self.n_steps):
            v_res = - vi * self.dtau # resistance to movement
            v_rand = dW[step]
            v_chem = self.delta * self.Gradient.calculate_gradient(xi) * \
                np.sin(phi/2) * self.dtau
                
            # print(v_chem)
            vi = vi + v_res + v_rand + v_chem
            xi = xi + vi * self.dtau
            x_history[step + 1, :] = xi
            v_history[step + 1, :] = vi
            xi_1 = x_history[step,:]
            theta = self.theta_ang(xi, xi_1)
            phi = self.phi_ang(xi, self.xa, theta)
            # print(np.degrees(theta))
        
        return x_history, v_history
            
        
            
    def theta_ang(self, xi, xi_1):
        x_axis = np.array([1,0])
        x_dir = xi - xi_1
        x_dir_norm= np.linalg.norm(x_dir)
        num = np.dot(x_axis, x_dir)
        
        den = x_dir_norm
        
        theta = np.acos(
            np.round(num / den, decimals= 10)
            )
        
        
        return theta 
    
             
             
    def phi_ang(self, xi, xa, theta):
        
        num = ((xa[0] - xi[0]) * np.cos(theta) + (xa[1] - xi[0]) * np.sin(theta))
        den = ((xa[0] - xi[0]) ** 2 + (xa[1] - xi[1]) ** 2) ** 1/2
        phi = np.acos(
            np.round(num/den, decimals = 10)
            )
        return phi
    
    def plot_sprouts(self):
        fig, ax = plt.subplots(figsize = (10,10), dpi = 300)
        keyshu = [x for x in self.x_storage.keys()]
        np.random.shuffle(keyshu)
        for key in keyshu:
            sprout = self.x_storage[key]
            ax.plot(sprout[:,0],sprout[:,1])
            ax.scatter(sprout[-1, 0], sprout[-1,1])
            
        

#%% Main body 
if __name__ == "__main__":
    n_reps = 100
    Hurst_index = 0.75
    n_steps = 10000
    dtau = 0.01
    delta = 0.5
    A_sim = AngioSimulation(n_reps, Hurst_index, n_steps, dtau, delta,
                            xa = [0, 1000])
    A_sim.simulate()
    A_sim.plot_sprouts()