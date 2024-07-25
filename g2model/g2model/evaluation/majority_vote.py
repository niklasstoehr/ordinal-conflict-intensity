
from g0configs import configs, helpers
from g1data import dataprepping, dataloading

import torch
import copy


def get_majority_hat(data, data_length = None, print_params = False):

    hat_majority = {"data":dict()}

    if data_length == None:
        data_length = list(data["data"].values())[0].shape
    else:
        data_length = list(data_length["data"].values())[0].shape

    ## evaluation
    for k in data["data"].keys():
        v = data["data"][k]
        if data["data"][k].dtype == torch.float32:
            hat_mean = torch.mean(v, dim=0)
            hat = hat_mean.repeat(data_length)
            if print_params:
                print(f"mean value: {str(hat_mean.item())}")

        elif data["data"][k].dtype == torch.int64:
            hat_mode = torch.mode(v, dim=0)[0]
            hat = hat_mode.repeat(data_length)
            if print_params:
                print(f"mode value: {str(hat_mode.item())}")
        hat_majority["data"][k] = hat

    return hat_majority




def get_majority_params(m, data, num_samples=1, epsilon=.0001):

    majority = get_majority_hat(data)
    majority_params = dict()
    n_c = 1

    for k in majority["data"].keys():
        majority_value = majority["data"][k][0].float()
        majority_params["lam_" + k + "_c"] = majority_value.repeat(n_c * num_samples).view(num_samples, n_c)

        for i in range(1, n_c):  ## add ordinal transform
            majority_params["lam_" + k + "_c"][:, i] = majority["data"][k][0] + (epsilon * (i))

    ## take care of special params
    majority_params["pi_Z_c"] = torch.tensor([1.0]).repeat(n_c * num_samples).view(num_samples, n_c)
    majority_params["pi_Z_c"][:, 1:] = 0.0

    majority_params["beta_Q_c"] = torch.tensor([0.0001]).repeat(n_c * num_samples).view(num_samples, n_c)
    majority_params["lam_Q_c"] = torch.tensor([0.9999]).repeat(n_c * num_samples).view(num_samples, n_c)
    majority_params["beta_probs_Q_c"] = torch.tensor([0.0001]).repeat(n_c * num_samples).view(num_samples, n_c)

    majority_params["std_G_c"] = torch.tensor([1.0]).repeat(n_c * num_samples).view(num_samples, n_c)
    #majority_params["lam_G_c"] = torch.tensor([0.8]).repeat(n_c * num_samples).view(num_samples, n_c)

    majority_params = {k: v.unsqueeze(-2) for k, v in majority_params.items()}

    ## initialise majority vote model
    m_major = copy.deepcopy(m)
    m_major.n_c = n_c

    return majority_params, majority, m_major






if __name__ == "__main__":

    config = configs.ConfigBase()
    df = dataloading.load_navco(csv_name="navco_mock.csv", G_cat=1, Q_thresh=0.3, Q_log=False, T_rank=True)
    train_data, test_data, df = dataprepping.prepare_df(df)
    majority_params = get_majority_params(train_data, num_samples=1, n_c=4)




