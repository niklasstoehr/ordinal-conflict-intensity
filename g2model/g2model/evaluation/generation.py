
import torch
from g2model.evaluation import trace_handlers


def get_empty_data(n_x=100, kwargs=None):

    no_data = {"G": None,
               "Q": None,
               "T": None,
               "A": None}

    data_mask = {"G": torch.ones(n_x).bool(),
                "Q": torch.ones(n_x).bool(),
                "T": torch.ones(n_x).bool(),
                "A": torch.ones(n_x).bool()}

    empty_data = {"data": no_data, "mask": data_mask}
    # "n_k": {"G": kwargs["n_G_k"], "Q": kwargs["n_Q_k"], "T": kwargs["n_T_k"]}}  ## specify n_k
    return empty_data



def generate_data(m, kwargs, params_samples = 100, n_x = 100, infer = -1):

    empty_data = get_empty_data(n_x)

    if isinstance(params_samples, dict):
        trace = trace_handlers.generate_trace(m, empty_data, params_samples= params_samples, cond="cond", infer=infer)

    elif isinstance(params_samples, int):
        trace = trace_handlers.generate_trace(m, empty_data, params_samples=params_samples, cond="uncond", infer=infer)

    accepted_sites = ['Z', 'Z_G', 'Z_Q', 'Z_T', 'G', 'Q', 'T', 'A']

    params = dict()
    sites = dict()

    for k in trace.nodes.keys():
        if k in accepted_sites:
            sites[k] = trace.nodes[k]["value"]
        else:
            if "value" in trace.nodes[k].keys() and k not in ['_RETURN', '_INPUT', 'data_plate']:
                params[k] = trace.nodes[k]["value"].view(1,1,-1)

    params = trace_handlers.pool_posterior(params)
    sites = trace_handlers.pool_posterior(sites)

    empty_data["data"] = sites
    return params, empty_data