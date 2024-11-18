
from g2model.models.model_types import gqta, a, g, q, t


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



