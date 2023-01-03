from g0configs import configs
from g1data import dataloading
from g3analysis import analysis

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

Adark = np.array([12, 18, 194]) / 256.0
Gdark = np.array([133, 12, 194]) / 256.0
Qdark = np.array([194, 12, 133]) / 256.0
Tdark = np.array([12, 179, 194]) / 256.0

#mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})


def plot(corr, mask):

    fig, ax = plt.subplots(1, figsize=(8, 8), gridspec_kw={'hspace': 0.35})
    fontsize = 28

    with sns.axes_style("white"):
        ax = sns.heatmap(
            corr,
            mask=mask,
            vmin=-1, vmax=1, center=0,
            annot=True, fmt='.2f',
            annot_kws={"size": fontsize},
            # cmap=sns.diverging_palette(20, 220, n=200),
            # cmap=sns.color_palette("vlag_r", as_cmap=True),
            cmap=sns.color_palette("RdGy", as_cmap=True),
            square=True,
            cbar=True,
            cbar_kws={"shrink": .82},
            linewidths=.0
        )

        # sns.set(font_scale=3.5)
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0,
            horizontalalignment='right'
        );

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_ticks([-1., -0.5, 0.0, 0.5, 1.0])
        cbar.set_ticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=fontsize, horizontalalignment='center')
        ax.set_yticklabels(ax.get_xmajorticklabels(), fontsize=fontsize, verticalalignment='center')

        ## label colors
        color_dict = {'G': Gdark, "G'": Gdark, 'Z': "black", "Z'": "black", 'Q': Qdark, 'gtrends': "black"}

        for tick_label in ax.get_xticklabels():
            key = tick_label.get_text()
            tick_label.set_color(color_dict[key])

        for tick_label in ax.get_yticklabels():
            key = tick_label.get_text()
            tick_label.set_color(color_dict[key])

        plt.show()
        return fig



def create_plot(df, query_name = ""):

    df_change = df.copy(deep=True)
    df_change["Z_diff"] = df_change["Z"].diff()
    df_change["G_diff"] = df_change["G"].diff()
    df_change = df_change.rename(columns={"Z": "Z", "Z_diff": "Z'", "gtab_" + str(query_name): "gtrends", "G_diff": "G'"})
    df_change = df_change[['G', "G'", 'Z', "Z'", 'Q', 'gtrends']]
    df_change = df_change.drop(columns =["Z'","G'"])
    corr = df_change.corr()

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False  ### want diagonal elements as well

    fig = plot(corr, mask)
    config = configs.ConfigBase()
    fig.savefig(config.get_path("plots") / Path("correlations_" + str(query_name) + ".pdf"), dpi=200)




if __name__ == "__main__":

    geo_date = {"geo": ["syria"], "date": ["2004-01-01", "2021-01-01"]}
    df = dataloading.load_navco(csv_name="navco_full.csv", geo_date=geo_date, G_cat=-1, Q_thresh=1.0, Q_log=True, T_rank=0)

    gtab_query = "syria"
    analysis_df = analysis.prepare_df(model_name="model_gqta_nc_7", gtab_query=[gtab_query], df=df, gtab_cat = 1209)
    create_plot(analysis_df, query_name=gtab_query)
