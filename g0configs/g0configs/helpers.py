import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import torch
from matplotlib.pyplot import cm
import matplotlib as mpl

from g0configs import configs

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def set_random_seed(random_seed):

    if random_seed == None:
        random_seed = torch.randint(0, 1000000, (1,)).item()
    return random_seed



def visualise_data_histogram(data):

    for k, v in data["data"].items():
        counted_data = Counter(v)

        fig,ax = plt.subplots(figsize=(8,3))
        sns.distplot(
            list(
                counted_data.keys()
            ),
            bins = np.arange(v.min(), v.max() + 1),
            hist_kws={
                "weights": list(counted_data.values())
            },
            kde = False
        )
        ax.set(xlim=(0,v.max()))
        ax.set(title=f"True {k}")

    plt.show()



def plot_single_distribution(data_mask_dict, site=None, n_bins=10, bin_range=None, log=False, label='Data'):

    titlefont = 14
    fontsize = 14
    labelsize = 14

    if site != None:
        data = data_mask_dict["data"][site].detach().numpy()
        mask = data_mask_dict["mask"][site].detach().numpy()
        data = data[mask]
    else:
        data = data_mask_dict.detach().numpy()

    if bin_range == None:
        bins = np.arange(n_bins)
        n, bins, patches = plt.hist(data, bins=bins, log=log, density=True, label=label)
    else:
        bins = np.linspace(bin_range[0], bin_range[1], num=n_bins)
        n, bins, patches = plt.hist(data, log=log, bins=bins, range=(0, len(bins)), align='left')

    plt.title(label, loc="left", fontsize=titlefont, color="black")
    plt.ylabel('prob', fontsize=fontsize, color="black")
    plt.xlabel(label, fontsize=fontsize, color="black")
    plt.tick_params(axis='both', which='major', labelsize=labelsize)




def store_model(m = None, posterior_params = None):

    config = configs.ConfigBase()
    path_name = config.get_path("models") / Path(str(m.model_type) + "_nc_" + str(m.n_c) + ".pkl")
    with open(path_name, 'wb') as f:
        model_params = {"m": m, "posterior_params": posterior_params}
        pickle.dump(model_params, f)
    print(f"stored model_params to {path_name}")



def load_model(file_name="model_gqta_nc_4"):

    config = configs.ConfigBase()
    path_name = config.get_path("models") / Path(str(file_name) + ".pkl")
    with open(path_name, 'rb') as f:
        model_params = pickle.load(f)
    print(f"loaded model_params from {path_name}")
    return model_params


def store_df(df=None, path_name = "latents", file_name = "df"):

    config = configs.ConfigBase()
    path_name = config.get_path(path_name) / Path(file_name + ".csv")
    df.to_csv(path_name, index=False)
    print(f"stored annotations to {path_name, file_name}")


def load_df(path_name = "latents", file_name="nx_33625_nc_4_geo_all_time_all"):

    config = configs.ConfigBase()
    path_name = config.get_path(path_name) / Path(str(file_name) + ".csv")
    df = pd.read_csv(path_name)
    print(f"loaded annotations from {path_name, file_name}")
    return df


