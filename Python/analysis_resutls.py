# %%
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import matplotlib.cm as cm
def batch_histogram_delta(folder: str,
                    H: float = None)->None:
    """
    Plots the histogram using a gaussian kernel density estimation 
    - The constant value is the Hurst parameter and the changing is based on the delta parameter
    
    """
    linspa = np.linspace(0, 100, 1000)
    fig, ax = plt.subplots(figsize = (15,5))
    df_to_hist = pd.DataFrame(columns=['H', 'Delta', 'Array'])
    index = 0
    for file in os.listdir(folder):
        file_name = join(folder, file)
        delta = str(file.split('eq')[1][1:].replace('_','.'))
        data = pd.read_csv(file_name, header = None).to_numpy().reshape(-1,)
        df_to_hist.at[index, 'H'] = H
        df_to_hist.at[index,'Delta'] = delta
        df_to_hist.at[index,'Array'] = data
        index += 1
    df_to_hist.sort_values(by = 'Delta', inplace=True)
    df_to_hist.reset_index(inplace=True, drop = True)
    # print(df_to_hist)
    color_map = plt.get_cmap('coolwarm', len(df_to_hist))
    for i in df_to_hist.index:
            
        data = df_to_hist.at[i, 'Array']
        delta = df_to_hist.at[i, 'Delta']
        non_nan_data = data[~np.isnan(data)]
        #print(non_nan_data)
        por_nan = (len(data) - len(non_nan_data))/len(data)
        kde = stats.gaussian_kde(non_nan_data)
        str_label = f'Delta = {float(delta):.2f}; {por_nan:.2f} non-reaching'
        ax.plot(linspa, kde(linspa), label = str_label, color = color_map(i)) 
    ax.legend()
    ax.set_title(f'H : {H}')
    plt.show()
    
    
def batch_histogram_hurst(folder: str,
                    Delta: float = None)->None:
    """
    Plots the histogram using a gaussian kernel density estimation 
    - The constant value is the Hurst parameter and the changing is based on the delta parameter
    
    """
    linspa = np.linspace(0, 100, 1000)
    fig, ax = plt.subplots(figsize = (15,5))
    df_to_hist = pd.DataFrame(columns=['H', 'Delta', 'Array'])
    index = 0
    for file in os.listdir(folder):
        file_name = join(folder, file)
        h = str(file.split('eq')[1][1:].replace('_','.'))
        data = pd.read_csv(file_name, header = None).to_numpy().reshape(-1,)
        df_to_hist.at[index, 'H'] = h
        df_to_hist.at[index,'Delta'] = Delta
        df_to_hist.at[index,'Array'] = data
        index += 1
    df_to_hist.sort_values(by = 'H', inplace=True)
    df_to_hist.reset_index(inplace=True, drop = True)
    # print(df_to_hist)
    color_map = plt.get_cmap('coolwarm', len(df_to_hist))

    for i in df_to_hist.index:
            
        data = df_to_hist.at[i, 'Array']
        hurst = df_to_hist.at[i, 'H']
        non_nan_data = data[~np.isnan(data)]
        #print(non_nan_data)
        por_nan = (len(data) - len(non_nan_data))/len(data)
        kde = stats.gaussian_kde(non_nan_data)
        str_label = f'H = {float(hurst):.3f}; {por_nan:.2f} non-reaching'
        ax.plot(linspa, kde(linspa), label = str_label, color = color_map(i)) 
    ax.legend()
    ax.set_title(f'Delta : {Delta}')    
    ax.legend(bbox_to_anchor = (1.3, 1))
    plt.show()
#%%
if __name__ == '__main__':
    batch_histogram_delta('delta_change_035_from_cluster', 
                          H = 0.35)
    batch_histogram_delta('delta_change_05_from_cluster',
                          H = 0.5)
    batch_histogram_delta('delta_change_065_from_cluster',
                        H = 0.65)
    batch_histogram_hurst('h_change_all_from_cluster',
                          Delta = 1)
    batch_histogram_hurst('h_change_close_from_cluster',
                          Delta = 1)
    batch_histogram_hurst('h_change_closer_from_cluster',
                          Delta = 1)
# %%
