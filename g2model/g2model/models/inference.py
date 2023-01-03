import pyro
import torch
from pyro.infer import NUTS, MCMC
from pyro.infer.autoguide import initialization as mcmc_inits
from pyro.infer import Predictive

from g0configs import helpers
from g2model.evaluation import trace_handlers



def run_mcmc(m, data, kwargs, print_params = True):

    #print(f"model n_c: {m.n_c}, n_samples: {n_samples}, n_warmup: {n_warmup}")
    if print_params:
        print(f"n_samples: {kwargs['n_samples']}, n_warmup: {kwargs['n_warmup']}")

    pyro.clear_param_store()
    torch.manual_seed(helpers.set_random_seed(kwargs["random_seed"]))

    init_params = get_model_prior(m, data = data, kwargs = kwargs, print_params = print_params)
    nuts_kernel = NUTS(m.model, target_accept_prob=kwargs["accept_prob"], max_tree_depth= kwargs["nuts_tree"],
                       adapt_step_size=True, init_strategy = mcmc_inits.init_to_value(values = init_params, fallback = mcmc_inits.init_to_sample))

    mcmc = MCMC(nuts_kernel, num_samples= kwargs["n_samples"], warmup_steps= kwargs["n_warmup"], num_chains=1, mp_context="spawn")
    mcmc.run(data)

    if print_params:
        mcmc.summary(prob=0.75)
    return mcmc



def get_model_prior(m, data, kwargs, print_params=False):
    p = Predictive(m.model, num_samples=kwargs["init_n"])
    init_params = p(data)
    init_params = {k: torch.mean(v.float(), dim=0) for k, v in init_params.items() if
                   k not in ["Z", "G", "Q", "T", "A"]}

    t_unique, t_counts = torch.unique(data["data"]["T"], return_counts=True)
    t_init = t_counts / torch.sum(t_counts)
    init_params["pi_T_ck"] = t_init.repeat(1, m.n_c, 1)

    a_unique, a_counts = torch.unique(data["data"]["A"], return_counts=True)
    a_init = a_counts / torch.sum(a_counts)
    init_params["pi_A_ck"] = a_init.repeat(1, m.n_c, 1)

    if print_params:
        print(f"init_params: {init_params}")
    return init_params



def get_posterior(mcmc, n_samples=100):

    mcmc_params = mcmc.get_samples()
    mcmc_params_n_samples = mcmc_params["pi_Z_c"].shape[0]
    params = dict()

    for k in mcmc_params.keys():
        if n_samples <= mcmc_params_n_samples: ## use mcmc samples
            indeces = torch.arange(0,mcmc_params_n_samples, 1).long()
        elif n_samples > mcmc_params_n_samples: ## oversample
            indeces = torch.randint(0, mcmc_params_n_samples, (n_samples,))
        params[k] = mcmc_params[k][indeces, :]
    #params = {k: v.unsqueeze(-2) for k, v in params.items()}
    return params



def get_random(m, data, num_samples = 100, infer = -1, get_avg = False):

    trace = trace_handlers.generate_trace(m, data, params_samples= num_samples, cond="uncond", infer=infer)
    accepted_sites = ['Z', 'Z_G', 'Z_Q', 'Z_T', 'G', 'Q', 'T', 'A']

    sites = dict()
    params = dict()

    for k in trace.nodes.keys():
        if k in accepted_sites:
            sites[k] = trace.nodes[k]["value"]
        else:
            if "value" in trace.nodes[k].keys() and k not in ['_RETURN', '_INPUT', 'data_plate']:
                params[k] = trace.nodes[k]["value"]

    if get_avg:
        params = trace_handlers.pool_posterior(params)
        sites = trace_handlers.pool_posterior(sites)

    sites = {"data": sites}
    return params, sites


