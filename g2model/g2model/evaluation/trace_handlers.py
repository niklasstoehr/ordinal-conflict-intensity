import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import infer_discrete



def generate_trace(m, data, params_samples= None, cond="cond", infer=-1):

    ## 1 sample plate _____________________

    if isinstance(params_samples, dict):
        num_samples = list(params_samples.values())[0].shape[0]
        params = params_samples

    if isinstance(params_samples, int):
        num_samples = params_samples
        params = None

    sample_plate = pyro.plate("samples", num_samples, dim=-2)
    model = sample_plate(m.model)

    ## 2 conditioning ____________________

    if cond == "cond" and params != None:
        model = poutine.condition(model, params)

    elif cond == "uncond" and params == None:
        model = poutine.uncondition(model)  ## indepedent from obs

    ## 3 infer discrete_____________________

    if infer >= 0:
        ## 0: Viterbi-like MAP inference, Defaults to 1 (sample)
        model = infer_discrete(model, first_available_dim=-3, temperature=infer)

    trace = poutine.trace(model).get_trace(data)
    trace.compute_log_prob()

    return trace




def pool_posterior(sample_dict, pool_type = {}):

    sample_dict_avg = dict()
    sample_dict_conf =  dict()

    for k in sample_dict.keys():
        k_tensor = sample_dict[k]

        if len(k_tensor.shape) == 1:
            sample_dict_avg[k] = k_tensor ## nothing to average
            sample_dict_conf[k] = torch.zeros(k_tensor.shape[0])
        else:

            if len(k_tensor.shape) == 2:
                k_tensor_len = 2
                is_decimal = float(k_tensor[0][0].detach().item()) % 1
            elif len(k_tensor.shape) == 3:
                k_tensor_len = 3
                is_decimal = float(k_tensor[0][0][0].detach().item()) % 1

            if k in pool_type.keys():
                if pool_type[k] == "mean":
                    is_decimal = 1.0
                if pool_type[k] == "mode":
                    is_decimal = 0.0

            if is_decimal > 0:
                k_avg_tensor = torch.mean(k_tensor.float(), dim=0).float()
                k_conf_tensor = torch.std(k_tensor.float(), dim=0).float()
            else:
                k_avg_tensor = torch.mode(k_tensor.long(), dim=0)[0].long()
                k_conf_tensor = torch.std(k_tensor.float(), dim=0).float()

            if k_tensor_len == 2:
                sample_dict_avg[k] = k_avg_tensor.view(-1)
                sample_dict_conf[k] = k_conf_tensor.view(-1)
            elif k_tensor_len == 3:
                sample_dict_avg[k] = k_avg_tensor.view(1, 1, -1)
                sample_dict_conf[k] = k_conf_tensor.view(1, 1, -1)

    return sample_dict_avg, sample_dict_conf




def sites_to_tensor(trace, sites=["G", "Q", "T", "A", "Z"]):
    sites_log = dict()
    sites_data = dict()

    for s in trace.nodes.keys():
        if s in sites:
            sites_data[s] = trace.nodes[s]["value"]
            if "log_prob" in trace.nodes[s].keys():
                sites_log[s] = trace.nodes[s]["log_prob"]

    return sites_data, sites_log


