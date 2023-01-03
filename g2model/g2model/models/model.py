
from g2model.models.model_types import gqta, a, g, q, t
from g2model.evaluation import predictive, metrics


def get_model(model_type = "gqta", n_c = "4"):

    if model_type == "gqta":
        m = gqta.Model(n_c)

    if model_type == "a":
        m = a.Model(n_c)

    if model_type == "g":
        m = g.Model(n_c)

    if model_type == "q":
        m = q.Model(n_c)

    if model_type == "t":
        m = t.Model(n_c)

    return m



def evaluate(m, params, test_data, posterior_type):

    sites, loglik = predictive.make_prediction(params, m, test_data)

    metrics.compute_exp_pred_lik(loglik, posterior_type=posterior_type)
    metrics.evaluate_point_predictons(sites, test_data, posterior_type=posterior_type)
    metrics.evaluate_distr(sites, test_data, posterior_type=posterior_type)


