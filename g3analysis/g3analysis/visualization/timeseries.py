from g0configs import configs
from g1data import dataloading
from g3analysis import analysis

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import pearsonr
from datetime import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
import matplotlib.lines as mlines

Adark = np.array([12, 18, 194]) / 256.0
Gdark = np.array([133, 12, 194]) / 256.0
Qdark = np.array([194, 12, 133]) / 256.0
Tdark = np.array([12, 179, 194]) / 256.0


def add_coloured_gac(axis, x, y, linewidth, label):
    s = pd.Series(np.array(y), index=x)
    # convert dates to numbers first
    inxval = mdates.date2num(s.index.to_pydatetime())
    points = np.array([inxval, s.values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="RdYlGn_r", linewidth=linewidth)
    # set color to date values
    lc.set_array(y)
    axis.add_collection(lc)

    label = mlines.Line2D([], [], color='coral', alpha=0.9, label=label)
    return label, axis


def adjust_date(df, freq = 'W', start_date = '2004-01-01', end_date = '2020-01-01'):
#https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html

    df_changed = df.copy(deep = True)
    df_changed = df_changed.loc[start_date:end_date]
    df_changed = df_changed.groupby(pd.Grouper(freq=freq)).mean()
    return df_changed


def interpolate_series(x, y, steps = 4):
    x = np.arange(0, date.shape[0], steps)
    y = y[::steps]
    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, date.shape[0], 1)
    ynew = f(xnew)
    return ynew


def plot(date, Z, G, Q, gtab, Z_diff= None, G_diff = None):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(16, 12), gridspec_kw={'hspace': 0.35})

    fontsize = 12
    linewidth = 2

    ## figure 1 ___________

    Z_line, _ = add_coloured_gac(ax1, date, Z, linewidth, label="Z - latent")
    G_line, = ax1.plot(date, G, color=Gdark, linestyle="dotted", linewidth=linewidth, label="P - Goldstein")
    ax1.legend(handles=[Z_line, G_line], loc='upper left', fontsize=fontsize, framealpha=1.0, edgecolor="white")

    ax1.set_ylabel('intensity', fontsize=fontsize, color="black")
    ax1.tick_params(axis='y', labelsize=fontsize, colors="black", which='both')
    # ax1.spines['right'].set_alpha(0.0)
    ax1.text(0.05, 0.975, "Pearson: " + str(round(pearsonr(G, Z)[0], 3)), fontsize=fontsize, horizontalalignment='left',
             verticalalignment='bottom', transform=ax1.transAxes, zorder=3)

    ## figure 2 ___________

    ax2_2 = ax2.twinx()  # "grey"
    Z_line, _ = add_coloured_gac(ax2, date, Z, linewidth, label="Z - latent")
    Q_line, = ax2_2.plot(date, Q, color=Qdark, linestyle="dashdot", linewidth=linewidth, label="Q - victim counts")
    ax2_2.legend(handles=[Z_line, Q_line], loc='upper left', fontsize=fontsize, framealpha=1.0, edgecolor="white")

    ax2.set_ylim([min(Z) - 1, max(Z) + 1])
    ax2.set_ylabel('intensity', fontsize=fontsize, color="black")
    ax2.tick_params(axis='y', labelsize=fontsize, colors="black", which='both')

    ax2_2.spines['right'].set_visible(True)
    ax2_2.spines['right'].set_color(Qdark)
    ax2_2.set_ylabel('log-scaled victim counts', fontsize=fontsize, color=Qdark)
    ax2_2.tick_params(axis='y', labelsize=fontsize, colors=Qdark, which='both')
    ax2.text(0.05, 0.975, "Pearson: " + str(round(pearsonr(Q, Z)[0], 3)), fontsize=fontsize, horizontalalignment='left',
             verticalalignment='bottom', transform=ax2.transAxes, zorder=3)

    ## figure 3 ___________

    ax3_2 = ax3.twinx()
    G_diff_line, = ax3.plot(date, G, color=Gdark, linestyle="dotted", linewidth=linewidth,
                            label="P - Goldstein")
    gtab_line, = ax3_2.plot(date, gtab, color="grey", linestyle="dashed", linewidth=linewidth, label="Google Trends")
    ax3_2.legend(handles=[G_diff_line, gtab_line], loc='upper left', fontsize=fontsize, framealpha=1.0,
                 edgecolor="white")

    ax3.set_ylim([min(G) - 1, max(G) + 1])
    ax3.tick_params(axis='y', labelsize=fontsize, colors="black", which='both')
    ax3.set_ylabel('intensity', fontsize=fontsize, color="black")

    ax3_2.spines['right'].set_visible(True)
    ax3_2.spines['right'].set_color('grey')
    ax3_2.set_ylabel('Google trends', fontsize=fontsize, color="grey")
    ax3_2.tick_params(axis='y', labelsize=fontsize, colors="grey", which='both')
    ax3.text(0.05, 0.975, "Pearson: " + str(round(pearsonr(G, gtab)[0], 3)), fontsize=fontsize,
             horizontalalignment='left', verticalalignment='bottom', transform=ax3.transAxes, zorder=3)

    ## figure 4 ___________

    ax4_2 = ax4.twinx()
    Z_diff_line, _ = add_coloured_gac(ax4, date, Z, linewidth, label="Z - latent")
    gtab_line, = ax4_2.plot(date, gtab, color="grey", linestyle="dashed", linewidth=linewidth, label="Google Trends")
    ax4_2.legend(handles=[Z_diff_line, gtab_line], loc='upper left', fontsize=fontsize, framealpha=1.0,
                 edgecolor="white")

    ax4.set_ylim([min(Z) - 1, max(Z) + 1])
    ax4.tick_params(axis='y', labelsize=fontsize, colors="black", which='both')
    ax4.set_ylabel('intensity', fontsize=fontsize, color="black")

    ax4_2.spines['right'].set_visible(True)
    ax4_2.spines['right'].set_color('grey')
    ax4_2.set_ylabel('Google trends', fontsize=fontsize, color="grey")
    ax4_2.tick_params(axis='y', labelsize=fontsize, colors="grey", which='both')
    ax4.text(0.05, 0.975, "Pearson: " + str(round(pearsonr(Z, gtab)[0], 3)), fontsize=fontsize,
             horizontalalignment='left', verticalalignment='bottom', transform=ax4.transAxes, zorder=3)


    ax1.text(-0.05, 1.2, "A", fontsize=20, fontweight="bold", horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes)
    ax2.text(-0.05, 1.2, "B", fontsize=20, fontweight="bold", horizontalalignment='left', verticalalignment='top',
             transform=ax2.transAxes)
    ax3.text(-0.05, 1.2, "C", fontsize=20, fontweight="bold", horizontalalignment='left', verticalalignment='top',
             transform=ax3.transAxes)
    ax4.text(-0.05, 1.2, "D", fontsize=20, fontweight="bold", horizontalalignment='left', verticalalignment='top',
             transform=ax4.transAxes)

    plt.show()
    return fig



def create_plot(df, query_name = "", freq = "W", diff= False):

    df_changed = adjust_date(df, freq = freq)
    date = df_changed.index.to_numpy()

    G = df_changed["G"].to_numpy()
    Q = df_changed["Q"].to_numpy()
    Z = df_changed["Z"].to_numpy()

    gtab = df_changed["gtab_" + query_name].to_numpy()

    if diff:
        Z_diff = df_changed["Z"].diff().to_numpy()
        Z_diff[0] = 0.0
        #Z_diff = interpolate_series(x = date, y = Z_diff, steps = 4)
        G_diff = df_changed["G"].diff().to_numpy()
        G_diff[0] = 0.0
        #G_diff = interpolate_series(x = date, y = G_diff, steps = 4)
        #empty = np.repeat(np.array([-1.0]), date.shape)
        fig = plot(date, Z, G, Q, gtab, Z_diff, G_diff)
    else:
        fig = plot(date, Z, G, Q, gtab)

    config = configs.ConfigBase()
    fig.savefig(config.get_path("plots") / Path("time_series_" + str(query_name) + ".pdf"), dpi=200)




if __name__ == "__main__":

    geo_date = {"geo": ["yemen"], "date": ["2004-01-01", "2021-01-01"]}
    df = dataloading.load_navco(csv_name="navco_full.csv", geo_date = geo_date, G_range = [0.9999, 0.0001], Q_thresh=1.0, Q_log=True, T_rank=0)

    gtab_query = "iraq"
    analysis_df = analysis.prepare_df(model_name="model_gqta_nc_5", gtab_query=[gtab_query], df=df, gtab_cat = 1209)
    create_plot(analysis_df, query_name=gtab_query, freq="M")
