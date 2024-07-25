import torch
import pyro
from pyro.ops.indexing import Vindex
import pyro.distributions as dist
from pyro.distributions import transforms


class Model():

    def __init__(self, n_c):
        self.n_c = n_c
        self.model_type = "model_q"
        print(f"{self.model_type} n_c: {n_c}")

        if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def model(self, data=None):

        ## Prior_____________

        base_delta_Q_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * 1.0)
        delta_Q_c = pyro.sample("lam_Q_c", dist.TransformedDistribution(base_delta_Q_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))

        base_beta_Q_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * 1.0)
        beta_Q_c = pyro.sample("beta_Q_c", dist.TransformedDistribution(base_beta_Q_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))

        pi_Z_c = pyro.sample("pi_Z_c", dist.Dirichlet(torch.ones(self.n_c) / self.n_c))  ## translates into number of classes

        ## Likelihood ______________________________________

        with pyro.plate('data_plate', data["mask"]["Q"].shape[0]):

            Z = pyro.sample('Z', dist.Categorical(pi_Z_c), infer={"enumerate": "parallel"})
            Q = pyro.sample('Q', dist.ZeroInflatedDistribution(dist.Geometric(Vindex(torch.flip(beta_Q_c, [-1]))[..., Z.long()]),gate=Vindex(torch.flip(delta_Q_c, [-1]))[..., Z.long()]).mask(data["mask"]["Q"]), obs=data["data"]["Q"])

            return Z, Q


if __name__ == "__main__":
    pass

