import copy
import torch

from g2model.evaluation import trace_handlers


def get_imputed_data(data, sites=[]):
    imputed_data = copy.deepcopy(data)
    for k in sites:
        imputed_data['data'][k] = None
        imputed_data['mask'][k] = torch.zeros(imputed_data["mask"][k].shape[0]).bool()
    return imputed_data




def impute_infer_Z(impute_site = [], data = {}, posterior_samples = {}, m = None, infer = 1):

    ## infer P(Z | Q, T, A, theta)______

    imputed_data = get_imputed_data(data, sites=impute_site)

    imputed_trace = trace_handlers.generate_trace(m, imputed_data, posterior_samples, cond="cond", infer = infer)
    imputed_sites, _ = trace_handlers.sites_to_tensor(imputed_trace, sites=["Z"])
    fitted_params_Z = {**posterior_samples, **imputed_sites}

    return fitted_params_Z, imputed_data, imputed_trace




def make_prediction(posterior_samples, m, data, infer= 1, sites={"G": "mean", "Q": "mean", "T": "mode", "A": "mode"}):

    pooled_sites_dict = dict()
    sites_loglik_dict = dict()

    for site in sites.keys():

        ## infer P(Z | Q, T, A, theta)______
        fitted_params_Z, imputed_data, imputed_trace = impute_infer_Z([site], data, posterior_samples, m, infer = infer)
        imputed_site, _ = trace_handlers.sites_to_tensor(imputed_trace, sites=[site])

        ## infer P(G | Z, theta)______
        predictive_trace = trace_handlers.generate_trace(m, data, fitted_params_Z, cond ="cond", infer = -1)
        _, sites_loglik = trace_handlers.sites_to_tensor(predictive_trace, sites=[site])

        pooled_imputed_sites, pooled_imputed_conf = trace_handlers.pool_posterior(imputed_site, pool_type=sites)
        sites_loglik_dict = {**sites_loglik_dict, **sites_loglik}
        pooled_sites_dict = {**pooled_sites_dict, **pooled_imputed_sites}

    ## get Z______________

    imputed_sites, _ = trace_handlers.sites_to_tensor(imputed_trace, sites=["Z"])
    pooled_Z_site, pooled_Z_conf = trace_handlers.pool_posterior(imputed_sites, pool_type={"Z": "mean"})
    pooled_sites_dict = {**pooled_sites_dict, **pooled_Z_site}

    return pooled_sites_dict, sites_loglik_dict

