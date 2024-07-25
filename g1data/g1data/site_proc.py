
import re
import pandas as pd
import numpy as np

from g0configs import configs


## G___________________________

def resolve_goldstein_events(df, level="V10", if_nan = 0.0):

    config = configs.ConfigBase()
    e_map_df = pd.read_csv(config.get_path("mappings") / "goldstein_mappings.csv", index_col=None)

    if level == "V10":
        nonallowed_rows = df["V10"].astype(str).str.startswith("21")
        df = df[~nonallowed_rows]
        e_map_df = e_map_df[["action_code", "action_goldstein"]]
        e_map_df = e_map_df.drop_duplicates(subset="action_code")
        df = pd.merge(df, e_map_df, how='left', left_on="V10", right_on="action_code")
        df = df.drop(columns=["action_code"])
        df = df.rename(columns={"action_goldstein": "G"})

    if level == "V100":
        nonallowed_rows = df["V100"].astype(str).str.startswith("21")
        df = df[~nonallowed_rows]
        e_map_df = e_map_df[["event_code", "event_goldstein"]]
        e_map_df = e_map_df.drop_duplicates(subset="event_code")
        df = pd.merge(df, e_map_df, how='left', left_on="V100", right_on="event_code")
        df = df.drop(columns=["event_code"])
        df = df.rename(columns={"event_goldstein": "G"})
    df["G"] = df["G"].fillna(value=if_nan)
    return df



def transform_df_site(df, to_range = [5.0, 0.0], column_name="G", print_params = False):

    if len(to_range) == 2:
        if to_range[0] > to_range[1]:
            df[column_name] = (-1 * df[column_name]) ## flip direction of G
            lower = to_range[1]
            upper = to_range[0]
        else:
            lower = to_range[0]
            upper = to_range[1]

        ## transform range
        not_null = df[column_name].notnull()
        x = df[column_name][not_null].astype(float)
        df.loc[not_null, column_name] = ((((upper - lower) * (x - min(x))) / (max(x) - min(x))) + lower)

        if print_params:
            print(f"{column_name:7} unique: {len(df[column_name].value_counts())}, min:, {min(df[column_name])}, max: {max(df[column_name])}")
    return df



## Q___________________________

def split_number_string(row, if_nan = 0):
    if type(row["F"]) == str:
        nums = re.findall(r'\d+', row["F"])
        if len(nums) >= 1:
            num_F = int(nums[0])
        else:
            num_F = if_nan
    else:
        num_F = if_nan
    if type(row["I"]) == str:
        nums = re.findall(r'\d+', row["I"])
        if len(nums) >= 1:
            num_I = int(nums[0])
        else:
            num_I = if_nan
    else:
        num_I = if_nan
    return num_F, num_I



def resolve_quantifier(df, Q_thresh=1.0, Q_log=False):

    df[["F", "I"]] = df.progress_apply(lambda row: split_number_string(row), axis=1, result_type="expand")
    # df[["F", "I"]] = df[["F", "I"]].fillna(value=0.0)
    df["Q"] = df["F"].astype(int) + df["I"].astype(int)
    df = df.drop(columns=["F", "I"])

    #if isinstance(Q_thresh, int):
    #df = df[df["Q"] <= 1000]  ## remove long tail from Poisson
    if isinstance(Q_thresh, float) and abs(Q_thresh) < 1.0:
        df = df[df["Q"] > 0].append(df[df["Q"] == 0].sample(frac=min(abs(Q_thresh),1.0), replace=False), ignore_index=True)

    if abs(Q_thresh) == 2: ## make binary decision <--> fatality / no fatality
        df["Q"][df["Q"] > 0] = 1

    if Q_thresh < 0 and Q_log == False: ## reverse
        df["Q"] = df["Q"].max() - df["Q"]

    #Q = df["Q"].to_numpy().reshape(-1, 1)
    #qt_binner = KBinsDiscretizer(n_bins=Q_epsilon, encode="ordinal", strategy="quantile") ## discretize quantiles and bin
    #Q_trans = qt_binner.fit_transform(Q)
    #df["Q"] = Q_trans.astype(int)

    if Q_log:
        df["Q"] = df["Q"].astype(float)
        df["Q"] = np.log10(df["Q"] + 1.0)
        df["Q"] = np.ceil(df["Q"])#.astype(int)

    return df


## A, T___________________________

def check_actor_target(row, at3_at6_dict, A_or_T="T", if_nan="UNS"):

    if row[A_or_T + "3"] in at3_at6_dict.keys():
        return row[A_or_T + "3"], at3_at6_dict[row[A_or_T + "3"]]

    if row[A_or_T + "6"] in at3_at6_dict.keys():
        return row[A_or_T + "6"], at3_at6_dict[row[A_or_T + "6"]]

    else:  ## specify nan types
        return if_nan, at3_at6_dict[if_nan]


def resolve_actor_target(df, A_or_T="T", AT_rank=0, mapping="group4", ranking="ranking4"):

    config = configs.ConfigBase()
    at_map_df = pd.read_csv(config.get_path("mappings") / "actor_target_mappings.csv")

    at3_at6_dict = dict(zip(at_map_df.at3_at6_code, at_map_df[mapping]))
    df[[A_or_T + "Type", A_or_T]] = df.progress_apply(lambda row: check_actor_target(row, at3_at6_dict, A_or_T), axis=1,result_type="expand")
    df = df.drop(columns=[A_or_T + "3", A_or_T + "6"])

    ## for categorical actors types
    if AT_rank == 0:
        df[A_or_T] = pd.Categorical(df[A_or_T])
        df[A_or_T] = df[A_or_T].cat.codes.astype(int)

    ## for ranked actors types
    if AT_rank != 0:
        # print(df)
        at_rank_df = pd.read_csv(config.get_path("mappings") / "actor_target_rankings.csv")
        at3_at6_rank_dict = dict(zip(at_rank_df[mapping], at_rank_df[ranking]))
        df[A_or_T] = df.progress_apply(lambda row: at3_at6_rank_dict[row[A_or_T]], axis=1)
        df[A_or_T] = df[A_or_T].astype(int)
        if AT_rank <= -1:
            df[A_or_T] = df[A_or_T].max() - df[A_or_T]
    return df



## MAIN________________________________________

def prepare_sites(df, G_range = [5.0,  0.0], Q_thresh=1.0, Q_log = True, AT_rank=0):

    df = resolve_goldstein_events(df, level="V10")
    df = transform_df_site(df, to_range= G_range, column_name="G")
    df = resolve_actor_target(df, A_or_T="T", AT_rank=AT_rank, mapping="group4", ranking="ranking2")
    df = resolve_actor_target(df, A_or_T="A", AT_rank=0, mapping="group4")
    df = resolve_quantifier(df, Q_thresh, Q_log)
    return df



if __name__ == "__main__":

    config = configs.ConfigBase()
