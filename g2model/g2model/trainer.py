
from g0configs import helpers
from g1data import dataprepping, dataloading
from g2model.models import inference, model


def single_run(m, df, kwargs, print_params = True):

    train_data, test_data, _ = dataprepping.prepare_df(df, random_seed = kwargs["random_seed"], print_params = print_params)
    mcmc = inference.run_mcmc(m, train_data, kwargs, print_params)

    fitted_params = inference.get_posterior(mcmc, n_samples=kwargs["post_samples"])
    random_params, _ = inference.get_random(m, train_data, num_samples = fitted_params["pi_Z_c"].shape[0])

    model.evaluate(m, fitted_params, test_data, "fitted")
    model.evaluate(m, random_params, test_data, "random")
    helpers.store_model(m, fitted_params)



if __name__ == "__main__":

    kwargs = {"n_c":5, "n_samples": 50, "n_warmup": 50, "random_seed": 4, "init_n": 200, "nuts_tree": 7, "accept_prob": 0.8, "post_samples": -1}

    #{"geo": ["syria"], "date": ["2004-01-01", "2020-01-01"]}
    #{"geo": ["iraq"], "date": ["2004-01-01", "2010-01-01"]}
    #{"not_geo": ["pakistan"], "date": []}
    #{"geo": [], "date": []}
    df = dataloading.load_navco(csv_name="navco_full.csv", geo_date = {"geo": [], "date": []}, G_range = [0.9999, 0.0001], Q_thresh=1.0, Q_log=True, T_rank=0)
    #df = dataloading.load_navco(csv_name="navco_full.csv", geo_date = {"geo": [], "date": []}, G_range = [0.9999, 0.0001], Q_thresh=1.0, Q_log=True, T_rank=0)
    m = model.get_model(model_type = "gqta", n_c = kwargs["n_c"])
    single_run(m, df, kwargs)

