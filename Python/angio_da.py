#%%
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as distance
import scipy.special as special
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import matplotlib.cm as cm
import re
import pandas as pd
from tqdm import tqdm
import gc
from joblib import Parallel, delayed

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


def js_distance_dict(dict_exp, experiment, h1_compare):
    '''
    Generates the KL distance for the wall and the h1_compare
    and the rest of the values in the dictionary
    The dict_exp is a dictionary with the following structure:
    dict_exp[wall][h] = data 
        where h is the hurst value and data is the data
        and wall is the wall value
    The output is a dictionary with the following structure:
    dict_out[h2] = kl_distance
        where h2 is the hurst value and kl_distance is the KL distance of the test
    
    Parameters:
    dict_exp: dict
        dictionary with the data as descripted before
    wall: int
        wall value to compare
    h1_compare: float
        hurst value to compare
    '''
    dict_work = dict_exp[experiment].copy()
    get_h1 = dict_work[h1_compare]
    
    # clean h1 nan values
    get_h1 = get_h1[~np.isnan(get_h1)]
    n_samples = 0.1 * len(get_h1)
    
    # kde_h1 = stats.gaussian_kde(get_h1.iloc[:,0])
    dict_out = {}
    for h2 in dict_work.keys():
        if h2 == h1_compare:
            continue
        # clean h2 nan values
        get_h2 = dict_work[h2]
        get_h2 = get_h2[~np.isnan(get_h2)]
        
        bin_h2 = np.histogram(get_h2, bins = 200, density = True)
        bin_h1 = np.histogram(get_h1, bins = bin_h2[1], density = True)
        dict_out[h2] = distance.jensenshannon(bin_h1[0], bin_h2[0])
        # dict_out[h2] = distance.jensenshannon(get_h1, get_h2)
    return dict_out

def mann_witney_u_dict(dict_exp, experiment, h1_compare):
    '''
    Generates the Mann-Witney U test for the wall and the h1_compare
    and the rest of the values in the dictionary
    The dict_exp is a dictionary with the following structure:
    dict_exp[wall][h] = data 
        where h is the hurst value and data is the data
        and wall is the wall value
    The output is a dictionary with the following structure:
    dict_out[h2] = p_value
        where h2 is the hurst value and p_value is the p-value of the test
    
    Parameters:
    dict_exp: dict
        dictionary with the data as descripted before
    wall: int
        wall value to compare
    h1_compare: float
        hurst value to compare
    '''
    dict_work = dict_exp[experiment].copy()
    get_h1 = dict_work[h1_compare]
    # clean h1 nan values
    # get_h1 = get_h1[~np.isnan(get_h1)]
    dict_out = {}
    for h2 in dict_work.keys():
        # print(h2)
        if h2 == h1_compare:
            continue
        # clean h2 nan values
        get_h2 = dict_work[h2]
        get_h2 = get_h2[~np.isnan(get_h2)]
        if len(get_h1) > len(get_h2):
            get_h1 = get_h1.sample(len(get_h2))
        else:
            get_h2 = get_h2.sample(len(get_h1))
        # print(len(get_h1), len(get_h2))
        dict_out[h2] = stats.mannwhitneyu(get_h1, get_h2, alternative = 'two-sided', nan_policy='omit').pvalue
    return dict_out

    
def hellinger_distance(binh1, binh2):   
    '''
    return the hellinger distance between two kde
    Parameters:
    kde_1: scipy.stats.gaussian_kde
        kde 1
    kde_2: scipy.stats.gaussian_kde
        kde 2
    linspa: np.array
        linear space to evaluate the kde
    Comments:
    ---
    this is computed as:
    H(P, Q) = 1/sqrt(2) * sqrt(sum((sqrt(p_i) - sqrt(q_i))^2))
    
    where p_i and q_i are the values of the kde at the point i
    https://www.bayesia.com/bayesialab/key-concepts/hellinger-distance
    '''
    p_vals = binh1
    q_vals = binh2
    
    # Compute the Hellinger distance using vectorized operations
    sqrt_diff = np.sqrt(p_vals) - np.sqrt(q_vals)

    diff_sqr = np.linalg.norm(sqrt_diff)
    
    # Scale by the Hellinger factor
    return np.linalg.norm(sqrt_diff) / np.sqrt(2)


    
def hellinger_distance_dict(dict_exp, experiment, h1_compare):
    """_summary_
    Generate the hellinger distance between the h1_compare and the rest of the values
    in the dictionary given a certain experiment and a h to compare
    
    Parameters
    ----------
    dict_exp : dict
        dictionary with the data as descripted before
    experiment : str/int
        experiment to compare, the key of the dictionary
    h1_compare : float
        hurst value to compare

    Returns
    -------
    dict_out : dict
        dictionary with the hellinger distance between the h1_compare and the rest of the values
        in the dictionary
    """    
    dict_work = dict_exp[experiment].copy()
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
        bin_h2 = np.histogram(get_h2, bins = 200, density = True)
        bin_h1 = np.histogram(get_h1, bins = bin_h2[1], density = True)
        dict_out[h2] = hellinger_distance(bin_h1[0], bin_h2[0])
    return dict_out

def ks_pval_dict(dict_exp, experiment, h1_compare):
    dict_work = dict_exp[experiment].copy()
    
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

        dict_out[h2] = stats.ks_2samp(get_h1, get_h2, nan_policy = 'omit').pvalue
    return dict_out
def boostrap_median(population, n_samples):
    sample = np.random.choice(population, n_samples, replace = True)
    return np.median(sample)

def bt_median_ttest_pval(h1, h2, n_samples, n_cpu, n_boostrap):
    
    median_h1 = Parallel(n_jobs = n_cpu, backend='threading')(delayed(boostrap_median)(h1.values.flatten(), n_samples) for _ in range(n_boostrap))
    median_h2 = Parallel(n_jobs = n_cpu, backend='threading')(delayed(boostrap_median)(h2.values.flatten(), n_samples) for _ in range(n_boostrap))
    
    return stats.ttest_ind(median_h1, median_h2, equal_var = False, nan_policy = 'omit', alternative='two-sided').pvalue
    
    

def boostreap_median_ttest_dict(dict_exp, experiment, h1_compare, n_samples ,n_cpu, n_boostrap):
    dict_work = dict_exp[experiment].copy()
    
    get_h1 = dict_work[h1_compare]
    get_h1 = get_h1[~np.isnan(get_h1)]
    dict_out = {}
    
    for h2 in dict_work.keys():
        if h2 == h1_compare:
            continue
        get_h2 = dict_work[h2]
        get_h2 = get_h2[~np.isnan(get_h2)]
        dict_out[h2] = bt_median_ttest_pval(get_h1, get_h2, n_samples, n_cpu, n_boostrap)
    return dict_out
    

def plot_hellinger_distance(dict_exp, h1_compare):
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_exp[list(dict_exp.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)
    for key in tqdm(dict_exp.keys()):
        dict_hellinger = hellinger_distance_dict(dict_exp, key, h1_compare)
        # plot in the x axis the wall values and in the y axis the hellinger distance
        values_hell = np.array(list(dict_hellinger.values())).T
        # ax.boxplot(values_hell[0], positions = [key], widths = 0.5)
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
    

def plot_ks_pval(dict_exp, h1_compare, log = False, signif = 0.05):
    '''
    Plot the ks p-value between the h1_compare and the rest of the values
    in the dictionary given a certain experiment and a h to compare
    Parameters:
    dict_exp: dict
        dictionary with the data as descripted before
    h1_compare: float
        hurst value to compare
    log: bool
        if True the y axis is in log scale
    '''
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_exp[list(dict_exp.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)

    for key in tqdm(dict_exp.keys()):
        dict_ks = ks_pval_dict(dict_exp, key, h1_compare)

        # plot in the x axis the wall values and in the y axis the hellinger distance
        # print(values_ks)
        # asign a color to each h value:
        # order the dict_hellinger by the keys
        dict_ks = dict(sorted(dict_ks.items()))   
        for i, h in enumerate(dict_ks.keys()):
            # print(key, dict_ks[h])
            ax.scatter(key, dict_ks[h], color = color_map(i) )
    
    ax.legend(handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Hurst = {h}',
                                    markerfacecolor=color_map(i), markersize=10) for i, h in enumerate(dict_ks.keys())],
                                    title = 'Hurst', title_fontsize = 'small', fontsize = 'small',
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
            #   ,bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Kolmogorov-Smirnov between Hurst = {h1_compare} and the others')
    ax.set_xlabel('Wall distance [a.u.]')
    ax.set_ylabel('KS p-value')
    x_lims = ax.get_xlim()
    ax.hlines(signif, x_lims[0], x_lims[1], color = 'red', linestyle = '--')
    if log == True:
        ax.set_yscale('log')
    # ax.set_ylim([1e-5, 1])
    plt.show()
    
    
def plot_mwu_pval(dict_exp, h1_compare, log = False, signif = 0.05):    
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_exp[list(dict_exp.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)
    for key in tqdm(dict_exp.keys()):
        dict_mwu = mann_witney_u_dict(dict_exp, key, h1_compare)
        # plot in the x axis the wall values and in the y axis the hellinger distance
        # asign a color to each h value:
        # order the dict_hellinger by the keys
        dict_mwu = dict(sorted(dict_mwu.items()))   
        for i, h in enumerate(dict_mwu.keys()):
            ax.scatter(key, dict_mwu[h], color = color_map(i) )
    ax.legend(handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Hurst = {h}',
                                    markerfacecolor=color_map(i), markersize=10) for i, h in enumerate(dict_mwu.keys())],
                                    title = 'Hurst', title_fontsize = 'small', fontsize = 'small',
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
            #   ,bbox_to_anchor=(1.05, 1), loc='upper left')
    x_lims = ax.get_xlim()
    ax.hlines(0.05, x_lims[0], x_lims[1], color = 'red', linestyle = '--')
    ax.set_title(f'Mann-Whitney U between Hurst = {h1_compare} and the others')
    ax.set_xlabel('Wall distance [a.u.]')
    ax.set_ylabel('Mann-Whitney U p-value')
    if log == True:
        ax.set_yscale('log')
    plt.show()

def plot_js_distance(dict_exp, h1_compare, log = False, signif = 0.05):
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_exp[list(dict_exp.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)
    for  key in tqdm(dict_exp.keys()):
        dict_js = js_distance_dict(dict_exp, key, h1_compare)
        # plot in the x axis the wall values and in the y axis the hellinger distance
        # asign a color to each h value:
        # order the dict_hellinger by the keys
        dict_js = dict(sorted(dict_js.items()))   
        for i, h in enumerate(dict_js.keys()):
            ax.scatter(key, dict_js[h], color = color_map(i) )
        
    ax.legend(handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Hurst = {h}',
                                    markerfacecolor=color_map(i), markersize=10) for i, h in enumerate(dict_js.keys())],
                                    title = 'Hurst', title_fontsize = 'small', fontsize = 'small',
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
            #   ,bbox_to_anchor=(1.05, 1), loc='upper left')
    x_lims = ax.get_xlim()
    if signif != None:
        
        ax.hlines(signif, x_lims[0], x_lims[1], color = 'red', linestyle = '--')
    
    ax.set_title(f'Jensen-Shannon distance between Hurst = {h1_compare} and the others')
    ax.set_xlabel('Wall distance [a.u.]')
    ax.set_ylabel('JS distance')
    if log == True:
        ax.set_yscale('log')
    plt.show()
    

     
    
def plot_median_boostrap_ttest(dict_exp, h1_compare, log = False, signif = 0.05, n_cpu = 4, n_samples = 1000, n_boostrap = 200):    
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    len_h = len(dict_exp[list(dict_exp.keys())[0]].values())
    color_map = plt.get_cmap('coolwarm', len_h)
    for  key in tqdm(dict_exp.keys()):
        dict_btt = boostreap_median_ttest_dict(dict_exp, key, h1_compare, n_cpu, n_samples, n_boostrap )
        
        # plot in the x axis the wall values and in the y axis the hellinger distance
        # asign a color to each h value:
        dict_btt = dict(sorted(dict_btt.items()))   
        for i, h in enumerate(dict_btt.keys()):
            ax.scatter(key, dict_btt[h], color = color_map(i) )
    ax.legend(handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Hurst = {h}',
                                    markerfacecolor=color_map(i), markersize=10) for i, h in enumerate(dict_btt.keys())],
                                    title = 'Hurst', title_fontsize = 'small', fontsize = 'small',
                                    bbox_to_anchor=(1.05, 1), loc='upper left')
    del dict_btt
    gc.collect()
            #   ,bbox_to_anchor=(1.05, 1), loc='upper left')
    x_lims = ax.get_xlim()
    if signif != None:
        
        ax.hlines(signif, x_lims[0], x_lims[1], color = 'red', linestyle = '--')
    
    ax.set_title(f'Boostrap Median t-test between Hurst = {h1_compare} and the others')
    ax.set_xlabel('Wall distance [a.u.]')
    ax.set_ylabel('p-value')
    if log == True:
        ax.set_yscale('log')
    plt.show()
    
    
def sampling_statistic_distribution(population_data, statistic, sample_size):
    
    sample = np.random.choice(population_data, sample_size, replace = True) 
    return statistic(sample)   
    
    
def plot_sampling_statistic(dict_exp, experiment, statistic, n_cpu = 4, sample_size = 200, boostrap_size = 500,
                            n_bins = 50, kde_plot = True):
    dict_to_plot = dict_exp[experiment]
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    color_map = plt.get_cmap('coolwarm', len(dict_to_plot.keys()))
    color = 0 
    max_plot = 0
    dict_to_plot = dict(sorted(dict_to_plot.items()))
    for i in dict_to_plot.keys():
        data = dict_to_plot[i]
        non_nan_data = data[~np.isnan(data)].iloc[:,0]
        statistic_values = Parallel(n_jobs = n_cpu, backend='threading')(delayed(sampling_statistic_distribution)(non_nan_data, statistic, sample_size) for _ in range(boostrap_size))
        kde = stats.gaussian_kde(statistic_values, bw_method='silverman')
        min_kde = np.min(statistic_values)
        max_kde = np.max(statistic_values)
        linspa = np.linspace(0, max_kde, 100)
        str_label = f'Hurst = {float(i):.2f}'
        if kde_plot == True:
            ax.plot(linspa, kde(linspa), color = color_map(color), label = str_label)
        else:
            ax.hist(statistic_values, bins = n_bins, density = True, alpha = 0.5, color = color_map(color),
                    histtype=u'step', label = str_label)
        color += 1
        if max_kde > max_plot:
            max_plot = max_kde
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Sampling statistic: {statistic.__name__} Wall : {experiment}')
    ax.set_xlim([0, max_plot])
    plt.show()
    
    
def wall_plot(dict_exp, experiment ,xlims = [0,100],
                    kde_plot = True, plot_median = False, 
                    n_bins = 50)->None:
    
    dict_to_plot = dict_exp[experiment]
    # order the dictionary
    dict_to_plot = dict(sorted(dict_to_plot.items()))
    len_space = int(0.1*len(dict_to_plot[list(dict_to_plot.keys())[0]]))
    fig, ax = plt.subplots(figsize = (15,5), dpi = 600)
    color_map = plt.get_cmap('coolwarm', len(dict_to_plot.keys()))
    # print(len(dict_to_plot.keys()))
    color = 0
    max_plot = 0
    for i in dict_to_plot.keys():
        data = dict_to_plot[i]
        non_nan_data = data[~np.isnan(data)].iloc[:,0]
        # print(non_nan_data)
        por_nan = (len(data) - len(non_nan_data))/len(data)
        kde = stats.gaussian_kde(non_nan_data, bw_method='silverman')
        min_kde = np.min(non_nan_data)
        max_kde = np.max(non_nan_data)
        linspa = np.linspace(0, max_kde, len_space)
        # str_label = f'Hurst = {float(i):.2f}; {por_nan:.2f} non-reaching'
        str_label = f'Hurst = {float(i):.2f}'
        # ax.plot(linspa, kde(linspa), label = str_label, color = color_map(color))
        if kde_plot == True:
            ax.plot(linspa, kde(linspa), color = color_map(color), label = str_label)
        else: 
            ax.hist(non_nan_data, bins = n_bins, density = True, alpha = 0.5, color = color_map(color),
                    histtype=u'step', label = str_label)
        if plot_median == True:
            ax.axvline(np.median(non_nan_data), color = color_map(color), linestyle = '--')
        color += 1
        if max_kde > max_plot:
            max_plot = max_kde
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f'Wall : {experiment}')
    ax.set_xlim([0, max_plot])
    plt.show()
    
#%%
if __name__ == '__main__':

    # dict_hwall = read_wall_hurst('h_change_wall_hurts_from_cluster')
    # # delect the 0 key from the dictionary
    # del dict_hwall[0]
    folder = 'exponential_h_wall_cluster'
    exponential_data = os.listdir('exponential_h_wall_cluster')

    # print(exponential_data)   
    dict_exponential = {} 
    dict_exponential_reversed = {}
    for file in exponential_data:
        # initialize the dictionary of said H
        list_hitime = pd.read_csv(os.path.join(folder, file))
        split__file = file.split('__')
        h = split__file[0].split('_')[-1]
        wall = split__file[1].split('_')[-1].split('.csv')[0]
        
        h = round(float(h),3)
        wall = int(float(wall))
        
        if h not in dict_exponential.keys():
            dict_exponential[h] = {}
        if wall not in dict_exponential_reversed.keys():
            dict_exponential_reversed[wall] = {}
        dict_exponential[h][wall] = list_hitime.values
        mask = list_hitime.values < list_hitime.values.max()
        dict_exponential_reversed[wall][h] = list_hitime[mask]
    # print(dict_exponential)
    del dict_exponential_reversed[0]    
    # plot_ks_pval(dict_exponential_reversed, 0.5, log = True, signif = 0.05)   
    # plot_hellinger_distance(dict_exponential_reversed, 0.5)
    # plot_js_distance(dict_exponential_reversed, 0.5, signif=np.sqrt(np.log(2)))
    # plot_mwu_pval(dict_exponential_reversed, 0.5, log = False, signif = 0.05)
    plot_sampling_statistic(dict_exponential_reversed, 2, np.median, n_cpu = 5,sample_size = 200, boostrap_size = 500)
    # for i in dict_exponential_reversed.keys():
    #     # print(i)
    #     if i == 1:
    #         # dict_exponential_reversed[i] = dict(sorted(dict_exponential_reversed[i].items()))

    #         wall_plot(dict_exponential_reversed, i, kde_plot = True, plot_median = True)
    #     # wall_plot(dict_exponential_reversed, , xlims = [0, 100], kde_plot=True)
#   # %%
    
    
    
    # plt.show()
    # print(five)

# %%
 