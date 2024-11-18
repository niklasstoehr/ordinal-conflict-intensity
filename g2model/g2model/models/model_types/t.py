import torch
import pyro
from pyro.ops.indexing import Vindex
import pyro.distributions as dist
from pyro.distributions import transforms


class Model():

    def __init__(self, n_c):
        self.n_c = n_c
        self.model_type = "model_t"
        print(f"{self.model_type} n_c: {n_c}")

        if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def model(self, data=None):

        ## Prior_____________

        n_T_k = 4  # torch.unique(data["data"]["T"]).shape[0]
        pi_T_ck = pyro.sample("pi_T_ck", dist.Dirichlet(torch.ones(self.n_c, n_T_k) / n_T_k).to_event(1))

        pi_Z_c = pyro.sample("pi_Z_c", dist.Dirichlet(torch.ones(self.n_c) / self.n_c))  ## translates into number of classes

        ## Likelihood ______________________________________

        with pyro.plate('data_plate', data["mask"]["Q"].shape[0]):

            Z = pyro.sample('Z', dist.Categorical(pi_Z_c), infer={"enumerate": "parallel"})
            T = pyro.sample('T', dist.Categorical(Vindex(pi_T_ck)[..., Z.long(), :]).mask(data["mask"]["T"]),obs=data["data"]["T"])

            return Z, T


if __name__ == "__main__":
    pass

