
from g0configs import configs
from g1data.site_proc import prepare_sites

from tqdm import tqdm
import pandas as pd
tqdm.pandas()



## geo and date ___________________________


def filter_geo_date(df, geo_date = {"geo": [], "not_geo": [], "date": []}):

    df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d', errors='coerce')

    if geo_date != None:
        if "geo" in geo_date.keys():
            if len(geo_date["geo"]) > 0:
                df = df[df['geo'].isin(geo_date["geo"])]
        if "not_geo" in geo_date.keys():
            if len(geo_date["not_geo"]) > 0:
                df = df[~df['geo'].isin(geo_date["not_geo"])]
        if "date" in geo_date.keys():
            if len(geo_date["date"]) == 2:
                df = df.loc[(df['date'] >= geo_date["date"][0]) & (df['date'] < geo_date["date"][1])]
    return df



def rename_columns(df):

    rename_dict = {"id": "id", "country_name": "geo", "date": "date", "cameo_actor_3": "A3", "cameo_actor_6": "A6", "target_3": "T3", "target_6": "T6",\
                    "verb_10": "V10", "verb_100": "V100", "fatal_casu": "F", "injuries": "I"}


    df = df[df.columns.intersection(list(rename_dict.keys()))]
    df = df.rename(columns=rename_dict)
    return df



## MAIN _____________________________


def load_navco(csv_name="navco_mock.csv", preloaded_df=None, geo_date=None, G_range = [5.0, 0.0], Q_thresh=500, Q_log=False,T_rank=0):

    config = configs.ConfigBase()

    if isinstance(preloaded_df, pd.DataFrame) == False:
        preloaded_df = pd.read_csv(config.get_path("navco") / csv_name)

    preloaded_df = rename_columns(preloaded_df)
    preloaded_df = filter_geo_date(preloaded_df, geo_date)
    preloaded_df = prepare_sites(preloaded_df, G_range, Q_thresh, Q_log, T_rank)
    print(f"number of data points: {len(preloaded_df)}")
    return preloaded_df



def load_navco_raw(csv_name="navco_mock.csv", frac=0.3):

    config = configs.ConfigBase()
    df = pd.read_excel(config.get_path("navco") / "navco3-0full.xlsx")
    print(f"load_navco_raw n:{int(len(df) * frac)}")
    df = df.sample(int(len(df) * frac), replace=False)

    ## add id
    df["group_index"] = df.groupby(["country_name", "date"]).cumcount()
    df["id"] = df.progress_apply(lambda row: str(row["country_name"]) + "_" + row["date"].strftime('%Y_%m_%d') + "_n" + str(row["group_index"]),axis=1)
    df = df.drop(columns=["group_index"])

    df.to_csv(config.get_path("navco") / csv_name, index=False)
    df = load_navco(csv_name)
    return df



if __name__ == "__main__":

    df = load_navco_raw(csv_name = "navco_full.csv", frac=1.0)
    df = load_navco(csv_name="navco_full.csv", preloaded_df=None, geo_date={"geo": [], "date": []}, G_range = [0.9999, 0.0001], Q_thresh = 1.0, Q_log = True, T_rank = 0)




