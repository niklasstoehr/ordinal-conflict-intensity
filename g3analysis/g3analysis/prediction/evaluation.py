from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def stationarity_test(df, diff_type = "0"):
    for c in df.columns:
        x = df[c]
        adf_res = adfuller(x)
        kpss_res = kpss(x, regression='ct')
        print(f'diff_type: {diff_type}, {c}: adf p value: {round(adf_res[1],4)} (small), kpss p value: {kpss_res[1]} (high)')



def df_diff(df, diff_type="diff"):
    if diff_type == "pct_change":
        df = df.pct_change().dropna()
    elif diff_type == "diff":
        df = df.diff()#.dropna()
        df.iloc[0, :] = df.iloc[1, :]
    df = df.replace([np.inf, -np.inf], 0.0)
    return df


def find_model_order(df, maxlag=12, verbose=True):
    model = VAR(df)
    if isinstance(maxlag, int): maxlag = range(1, maxlag + 1)  ## can pass int or list
    aic_res = np.empty((len(maxlag),))
    bic_res = np.empty((len(maxlag),))

    for i, lag in enumerate(maxlag):
        result = model.fit(lag)
        aic_res[i] = result.aic
        bic_res[i] = result.bic
        if verbose:
            print(f'Lag order = {lag}, AIC = {round(aic_res[i], 3)}, BIC = {round(bic_res[i], 3)}')

    aic_min_i = np.argmin(aic_res)
    aic_lag = aic_min_i + 1
    bic_min_i = np.argmin(bic_res)
    bic_lag = bic_min_i + 1

    print(f'aic lag {aic_lag}: {aic_res[aic_min_i]}, bic lag {bic_lag}: {bic_res[bic_min_i]}')
    return aic_lag, bic_lag



def granger_test(data, maxlag=12, variables=["G", "Z", "Q"], test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[c, r]], maxlag=maxlag, verbose=False)
            if isinstance(maxlag, int): maxlag = range(1, maxlag + 1)  ## can pass int or list
            p_values = [round(test_result[i][0][test][1], 5) for i in maxlag]
            if verbose: print(f'X = {r}, Y = {c}, p values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_y' for var in variables]
    df.index = [var + '_x' for var in variables]

    print(df)
    return df

