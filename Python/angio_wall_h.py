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
    
    return dict_out

def get_ks_test(dict_hwall: dict, wall: int, h1_compare: float) -> dict:
    """
    Returns the p-value of the ks test between the h1 and h2
    """
    dict_out = {}
    dict_work = dict_hwall[wall].copy()
    get_h1 = dict_work[h1_compare]
    for h2 in dict_work.keys():
        get_h2 = dict_work[h2]
        
        ks_test = stats.ks_2samp(get_h1, get_h2, nan_policy='omit')
        dict_out[h2] = ks_test.pvalue
    return dict_out 
#%%
if __name__ == '__main__':

    dict_hwall = read_wall_hurst('h_change_wall_hurts_from_cluster')
    ks_dict = {}
    for wall in dict_hwall.keys():
        ks_dict[wall] = get_ks_test(dict_hwall, wall, 0.5)
    print(ks_dict)
    
# %%
