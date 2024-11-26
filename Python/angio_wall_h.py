#%%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import matplotlib.cm as cm
import re

def read_wall_hurst(folder: str)->dict:
    """
    Reads the files in the folder and returns a dataframe with the data
    """
    list_dir = os.listdir(folder)
    
    dict_out = {}
    
    for file in list_dir:
        file_name = join(folder, file)
        hurst = float(re.search(r"hurst_eq([0-9_]+)", file).group(1).replace("_", "."))
        wall = int(float((re.search(r"wall_eq([0-9_]+)", file).group(1).replace("_", "."))))
        data = pd.read_csv(file_name, header = None).to_numpy().reshape(-1,)
        # verify if the dict_out[wall] is already created
        if wall in dict_out.keys():
            dict_out[wall][hurst] = data
        else:
            dict_out[wall] = {hurst: data}
    
    # order the dictionary
    dict_out = dict(sorted(dict_out.items()))
    
    return dict_out

def mann_witney_u_dict(dict_wall, wall, h1_compare):
    dict_work = dict_wall[wall].copy()
    get_h1 = dict_work[h1_compare]
    # clean h1 nan values
    get_h1 = get_h1[~np.isnan(get_h1)]
    dict_out = {}
    for h2 in dict_work.keys():
        if h2 == h1_compare:
            continue
        # clean h2 nan values
        get_h2 = dict_work[h2]
        get_h2 = get_h2[~np.isnan(get_h2)]
        dict_out[h2] = stats.mannwhitneyu(get_h1, get_h2, alternative = 'two-sided').pvalue
    return dict_out

    
def hellinger_distance(kde_1, kde_2, linspa):   
    '''
    return the hellinger distance between two kde
    
    this is computed as:
    H(P, Q) = 1/sqrt(2) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2))
    
    where p_i and q_i are the values of the kde at the point i
    https://www.bayesia.com/bayesialab/key-concepts/hellinger-distance
    '''
    H_pq = 0
    for i in range(len(linspa)):
        p_i = kde_1(linspa[i])
        q_i = kde_2(linspa[i])
        H_pq += (np.sqrt(p_i) - np.sqrt(q_i))**2
    return 1/np.sqrt(2) * np.sqrt(H_pq)


    
def hellinger_distance_dict(dict_wall, wall, h1_compare):
    
    dict_work = dict_wall[wall].copy()
    get_h1 = dict_work[h1_compare]
    

    # clean h1 nan values
    get_h1 = get_h1[~np.isnan(get_h1)]
    
    linspa = np.linspace(0, 100, 1_000)
    # print(linspa[-1])
    
    kde_h1 = stats.gaussian_kde(get_h1)
    dict_out = {}
    for h2 in dict_work.keys():
        if h2 == h1_compare:
            continue
        # clean h2 nan values
        get_h2 = dict_work[h2]
        get_h2 = get_h2[~np.isnan(get_h2)]
        kde_h2 = stats.gaussian_kde(get_h2)
        dict_out[h2] = hellinger_distance(kde_h1, kde_h2, linspa)
    return dict_out

def ks_distance_dict(dict_wall, wall, h1_compare):
    dict_work = dict_wall[wall].copy()
    get_h1 = dict_work[h1_compare]
    # clean h1 nan values
    linspa = np.linspace(0, 100, 1_000)
    dict_out = {}
    for h2 in dict_work.keys():
        if h2 == h1_compare:
            continue
        # clean h2 nan values
        get_h2 = dict_work[h2]
        get_h2 = get_h2[~np.isnan(get_h2)]

        dict_out[h2] = stats.ks_2samp(get_h1, get_h2, nan_policy = 'omit').pvalue
    return dict_out

def plot_hellinger_distance(dict_wall, h1_compare):
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_wall[list(dict_wall.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)
    for key in dict_wall.keys():
        dict_hellinger = hellinger_distance_dict(dict_wall, key, h1_compare)
        # plot in the x axis the wall values and in the y axis the hellinger distance
        values_hell = np.array(list(dict_hellinger.values())).T
        ax.boxplot(values_hell[0], positions = [key], widths = 0.5)
        # asign a color to each h value:
        # order the dict_hellinger by the keys
        dict_hellinger = dict(sorted(dict_hellinger.items()))   
        for i, h in enumerate(dict_hellinger.keys()):
            ax.scatter(key, dict_hellinger[h], color = color_map(i) )
    
    ax.legend(handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Hurst = {h}',
                                    markerfacecolor=color_map(i), markersize=10) for i, h in enumerate(dict_hellinger.keys())],
                                    title = 'Hurst', title_fontsize = 'small', fontsize = 'small',
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
            #   ,bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Hellinger distance between Hurst = {h1_compare} and the others')
    ax.set_xlabel('Wall distance [a.u.]')
    ax.set_ylabel('Hellinger distance')
    plt.show()
    

def plot_ks_pval(dict_wall, h1_compare, log = False):
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_wall[list(dict_wall.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)
    for key in dict_wall.keys():
        dict_ks = ks_distance_dict(dict_wall, key, h1_compare)
        if key == 26:
            print(dict_ks)
        # plot in the x axis the wall values and in the y axis the hellinger distance
        values_ks = np.array(list(dict_ks.values())).T
        # print(values_ks)
        ax.boxplot(values_ks, positions = [key], widths = 0.5)
        # asign a color to each h value:
        # order the dict_hellinger by the keys
        dict_ks = dict(sorted(dict_ks.items()))   
        for i, h in enumerate(dict_ks.keys()):
            ax.scatter(key, dict_ks[h], color = color_map(i) )
    
    ax.legend(handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Hurst = {h}',
                                    markerfacecolor=color_map(i), markersize=10) for i, h in enumerate(dict_ks.keys())],
                                    title = 'Hurst', title_fontsize = 'small', fontsize = 'small',
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
            #   ,bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Kolmogorov-Smirnov between Hurst = {h1_compare} and the others')
    ax.set_xlabel('Wall distance [a.u.]')
    ax.set_ylabel('KS p-value')
    if log == True:
        ax.set_yscale('log')
    # ax.set_ylim([1e-5, 1])
    plt.show()
    
def wall_plot(dict_wall, wall,xlims = [0,100],
                    kde = True)->None:
    
    dict_to_plot = dict_wall[wall]
    # order the dictionary
    dict_to_plot = dict(sorted(dict_to_plot.items()))
    linspa = np.linspace(0, 100, 1000)
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    color_map = plt.get_cmap('coolwarm', len(dict_to_plot.keys()))
    # print(len(dict_to_plot.keys()))
    color = 0
    for i in dict_to_plot.keys():
        data = dict_to_plot[i]
        non_nan_data = data[~np.isnan(data)]
        por_nan = (len(data) - len(non_nan_data))/len(data)
        kde = stats.gaussian_kde(non_nan_data)
        str_label = f'Hurst = {float(i):.2f}; {por_nan:.2f} non-reaching'
        # ax.plot(linspa, kde(linspa), label = str_label, color = color_map(color))
        ax.hist(non_nan_data, bins = 100, density = True, alpha = 0.5, color = color_map(color),
                histtype=u'step', label = str_label)
        color += 1
    ax.legend()
    ax.set_title(f'Wall : {wall}')
    ax.set_xlim(xlims)
    plt.show()
    
#%%
if __name__ == '__main__':

    dict_hwall = read_wall_hurst('h_change_wall_hurts_from_cluster')
    # delect the 0 key from the dictionary
    del dict_hwall[0]
    # dict_hwall
    plot_ks_pval(dict_hwall, 0.5, log = True)
    plot_ks_pval(dict_hwall, 0.5, log = False)
# %%
