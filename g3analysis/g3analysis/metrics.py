import pandas as pd
from scipy import stats

from g0configs import configs
from g1data import dataloading


def compute_correlation(df = None, a = None, b = None, print_corr = False):

    if isinstance(df ,pd.DataFrame):
        a = df[a].to_numpy()
        b = df[b].to_numpy()

    spearm = stats.spearmanr(a, b, alternative='two-sided')
    spearm = round(spearm[0] ,4)

    pears = stats.pearsonr(a, b)
    pears = round(pears[0] ,4)

    if print_corr:
        print(f"spearm: {spearm}, pears: {pears}")

    return spearm, pears


if __name__ == "__main__":


    config = configs.ConfigBase()
    df = dataloading.load_navco()
    spearm, pears = compute_correlation(df=df, a="G", b="Q", print_corr = True)