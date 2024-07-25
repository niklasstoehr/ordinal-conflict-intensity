
from g0configs import helpers
from g1data import dataloading, dataprepping
from g3analysis import latents
from g3analysis.gtrends import gtab_data

import pandas as pd
import numpy as np


def perform_imputation(df, value_columns):

    for k, v in value_columns.items():
        if k in df.columns:
            for interpol in v:
                if isinstance(interpol, str):
                    df[k] = df[k].interpolate(method=interpol, fill_value='extrapolate',limit_direction='both')
                if isinstance(interpol, float) or isinstance(interpol, int):
                    df[k] = df[k].fillna(interpol)
    return df



def merge_gtab(df, df_q):

    df_merged = pd.merge(df, df_q, how = "left", left_index=True, right_index=True)
    return df_merged



def prepare_gtab_df(df, freq = 'D'):

    c_names = list(df.columns)
    value_columns = {c:['linear', 'nearest'] for c in c_names}
    df = df.resample(freq).mean()
    df = perform_imputation(df, value_columns)
    return df


def group_impute(df, freq = 'D'):

    value_columns = {'date': [None], 'Z': ['linear', 'nearest'], 'Z_std': ['linear', 'nearest'], 'G': ['linear', 'nearest'], 'Q': [0], 'T': [df["Q"].mean()], 'A': [df["Q"].mean()], 'V10': ['linear', 'nearest']}
    df = df.set_index(pd.DatetimeIndex(df['date']))
    df = df.resample(freq).mean()
    df = perform_imputation(df, value_columns)
    return df


def prepare_df(model_name = "model_gqta_nc_5", geo_date = {"geo": [], "date": []}, gtab_query = "", df = None, gtab_cat = 1209):

    model_params = helpers.load_model(file_name=model_name)

    if isinstance(df, pd.DataFrame) == False:
        df = dataloading.load_navco(csv_name="navco_full.csv", geo_date = geo_date, G_range = [0.9999,0.0001], Q_thresh=1.0, Q_log=True, T_rank=0)
    data, _, df = dataprepping.prepare_df(df, heldout_frac = 0.0, shuffle = False)

    df = latents.annotate_latent(model_params, data, df)
    df = dataloading.filter_geo_date(df, geo_date=geo_date)
    df = group_impute(df)

    ## gtab
    g = gtab_data.GTAB_Handler(gtab_cat = gtab_cat)

    if gtab_query == "":
        df_q = g.query(geo_date["geo"])
    else:
        df_q = g.query(gtab_query)

    df_q = prepare_gtab_df(df_q)

    df_merged = merge_gtab(df, df_q)
    return df_merged



if __name__ == "__main__":
    df = prepare_df(geo_date = {"geo": [""], "date": ["2004-01-01", "2020-01-01"]})
    print(df)
