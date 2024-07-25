
import torch
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()

from g0configs import configs, helpers
from g1data import dataloading



def add_ord_obs_n_k(data, ordinal_categeories = ["G","Q","T","A"], print_params = True):

    data["n_k"] = dict()
    for k, v in data["data"].items():
        if k in ordinal_categeories:
            v_notnan = ~torch.isnan(v) ## get nonnan
            n_k = len(torch.unique(v[v_notnan]))
            data["n_k"][k] = int(n_k)
            if print_params:
                print(f"{k} has number of classes {data['n_k'][k]}")
        #else:
        #    data["n_k"][k] = None
    return data



def train_test_indeces(n_x=1000, heldout_frac=0.2, shuffle = True):
    if shuffle:
        ran_inds = torch.randperm(n_x)
    else: ## do not shuffle
        ran_inds = torch.arange(0, n_x, 1)
    split_ind = int(len(ran_inds) * (1-heldout_frac))

    train_ind = ran_inds[:split_ind]  ## split
    test_ind = ran_inds[split_ind:]  ## split
    return train_ind, test_ind



def prepare_nan_masking(data_values, nan_value = 0.0):
    data_type = data_values.dtype
    data_values_nan = torch.isnan(data_values)
    corr_data_values = torch.where(data_values_nan, nan_value, data_values.double()).type(data_type)
    mask = torch.where(data_values_nan, 0.0, 1.0)  ## missing values are 0
    mask = mask.bool() ## make boolean
    return corr_data_values, mask



def prepare_data_masks(data, indeces, print_params = True):

    if print_params:
        print(f"prepare_data_masks {indeces[:10]}...")
    masked_data = defaultdict(dict)
    masked_data["data"] = dict()
    masked_data["mask"] = dict()

    for k, v in data["data"].items():
        data_v, mask_v = prepare_nan_masking(v[indeces])
        masked_data["data"][k] = data_v
        masked_data["mask"][k] = mask_v

        if "n_k" in data.keys(): ## copy over n_k
            if k in data["n_k"].keys():
                masked_data["n_k"][k] = data["n_k"][k]

    return masked_data



def pack_dicts(df, include_data = ["G", "Q", "T", "A"]):
    data = defaultdict(dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(device)
    for c in include_data:
        if df[c].dtype == int:
            data["data"][c] = torch.LongTensor(df[c]).to(device)
        if df[c].dtype == float:
            data["data"][c] = torch.FloatTensor(df[c]).to(device)
    return data



def prepare_df(df, heldout_frac = 0.5, shuffle = True, include_data = ["G", "Q", "T", "A"], random_seed = 0, print_params = True):

    torch.manual_seed(helpers.set_random_seed(random_seed))

    data = pack_dicts(df.copy(), include_data)
    data = add_ord_obs_n_k(data, ordinal_categeories=include_data, print_params = print_params)

    ## train, test
    train_ind, test_ind = train_test_indeces(len(df.copy()), heldout_frac= heldout_frac, shuffle = shuffle)
    train_data = prepare_data_masks(data, train_ind, print_params = print_params)

    if len(test_ind.shape) > 0:
        test_data = prepare_data_masks(data, test_ind, print_params = print_params)
    else:
        test_data = torch.empty()
    print(df)
    return train_data, test_data, df




if __name__ == "__main__":

    config = configs.ConfigBase()
    df = dataloading.load_navco(G_range=[5.0, 0.0], Q_thresh=2.0, Q_log=False, T_rank=0)
    train_data, test_data, df = prepare_df(df, print_params = True)
    print(df)
    print(train_data)


