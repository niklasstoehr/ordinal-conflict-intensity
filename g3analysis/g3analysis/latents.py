
import numpy as np
import torch
from g0configs import helpers
from g1data import dataprepping, dataloading, site_proc
from g2model.evaluation import trace_handlers



def infer_Z(m, data = {}, posterior_samples = {}, pool_type = None, infer = 1):

    ## infer P(Z | G, Q, T, A, theta)______
    trace = trace_handlers.generate_trace(m, data, posterior_samples, cond="cond", infer = infer)
    Z, _ = trace_handlers.sites_to_tensor(trace, sites=["Z"])
    pooled_Z, pooled_Z_conf = trace_handlers.pool_posterior(Z, pool_type = pool_type)
    return pooled_Z, pooled_Z_conf, trace



def annotate_latent(model_params, data, df, full_c_range = True):

    sites, conf_Z, _ = infer_Z(model_params["m"], data, model_params["posterior_params"], pool_type = {"Z": "mode"})

    df["Z"] = sites["Z"].detach().cpu().numpy()
    df["Z_std"] = conf_Z["Z"].detach().cpu().numpy()

    if full_c_range:
        df["Z"] = (df["Z"] - 0) / (model_params["m"].n_c - 1)
    df = site_proc.transform_df_site(df, to_range = [0.0, 1.0], column_name="Z", print_params = False)
    df = site_proc.transform_df_site(df, to_range = [0.0, 1.0], column_name="Z_std", print_params = False)
    df = site_proc.transform_df_site(df, to_range = [0.0, 1.0], column_name="G", print_params = False)
    return df


def main(model_file = "model_gqta_nc_5", geo_date = {"geo": [], "date": []}):

    model_params = helpers.load_model(file_name=model_file)

    df = dataloading.load_navco(csv_name="navco_full.csv", geo_date = geo_date, G_range=[0.9999, 0.0001], Q_thresh=1.0, Q_log=True, T_rank=0)
    data, _, df = dataprepping.prepare_df(df, heldout_frac = 0.0, shuffle = False)

    df= annotate_latent(model_params, data, df)
    helpers.store_df(df=df, path_name= "latents", file_name = "nc_" + str(model_params["m"].n_c))# + "_" + str(geo_date["geo"][0]))
    return df


if __name__ == "__main__":

    df = main()
    print(df)


