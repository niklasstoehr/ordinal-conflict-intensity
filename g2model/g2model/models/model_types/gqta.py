import torch
import pyro
from pyro.ops.indexing import Vindex
import pyro.distributions as dist
from pyro.distributions import transforms


class Model():

    def __init__(self, n_c):
        self.n_c = n_c
        self.model_type = "model_gqta"
        print(f"{self.model_type} n_c: {n_c}")

        if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def model(self, data=None):

        ## Prior_____________

        #base_G_c = dist.Normal(torch.ones(self.n_c) * (4 / self.n_c), torch.ones(self.n_c))
        base_G_c = dist.Normal(torch.ones(self.n_c) * -1.0, torch.ones(self.n_c) * 1.0)
        #lam_G_c = pyro.sample("lam_G_c", dist.TransformedDistribution(base_G_c, [transforms.OrderedTransform()]))
        #std_G_c = pyro.sample("std_G_c", dist.Gamma(torch.ones(self.n_c), torch.ones(self.n_c)).to_event(1))
        mean_G_c = pyro.sample("mean_G_c", dist.TransformedDistribution(base_G_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))
        conc_G_c = pyro.sample("conc_G_c", dist.Gamma(torch.ones(self.n_c), torch.ones(self.n_c)).to_event(1))

        base_delta_Q_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * 1.0)
        delta_Q_c = pyro.sample("delta_Q_c", dist.TransformedDistribution(base_delta_Q_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))
        base_beta_Q_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * 1.0)
        beta_Q_c = pyro.sample("beta_Q_c", dist.TransformedDistribution(base_beta_Q_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))

        n_T_k = 4  # torch.unique(data["data"]["T"]).shape[0]
        pi_T_ck = pyro.sample("pi_T_ck", dist.Dirichlet(torch.ones(self.n_c, n_T_k) / n_T_k).to_event(1))

        n_A_k = 4  # torch.unique(data["data"]["T"]).shape[0]
        pi_A_ck = pyro.sample("pi_A_ck", dist.Dirichlet(torch.ones(self.n_c, n_A_k) / n_A_k).to_event(1))

        pi_Z_c = pyro.sample("pi_Z_c", dist.Dirichlet(torch.ones(self.n_c) / self.n_c))  ## translates into number of classes

        ## Likelihood ______________________________________

        with pyro.plate('data_plate', data["mask"]["Q"].shape[0]):

            Z = pyro.sample('Z', dist.Categorical(pi_Z_c), infer={"enumerate": "parallel"})

            mean_G_c = Vindex(mean_G_c)[..., Z.long()]
            conc_G_c = Vindex(conc_G_c)[..., Z.long()] + 2.0

            G = pyro.sample('G', dist.Beta(conc_G_c * mean_G_c, conc_G_c * (1 - mean_G_c)).mask(data["mask"]["G"]),obs=data["data"]["G"])
            #G = pyro.sample('G', dist.Normal(Vindex(lam_G_c)[..., Z.long()], Vindex(std_G_c)[..., Z.long()]).mask(data["mask"]["G"]), obs=data["data"]["G"])
            Q = pyro.sample('Q', dist.ZeroInflatedDistribution(dist.Geometric(Vindex(torch.flip(beta_Q_c, [-1]))[..., Z.long()]),gate=Vindex(torch.flip(delta_Q_c, [-1]))[..., Z.long()]).mask(data["mask"]["Q"]), obs=data["data"]["Q"])
            T = pyro.sample('T', dist.Categorical(Vindex(pi_T_ck)[..., Z.long(), :]).mask(data["mask"]["T"]),obs=data["data"]["T"])
            A = pyro.sample('A', dist.Categorical(Vindex(pi_A_ck)[..., Z.long(), :]).mask(data["mask"]["A"]),obs=data["data"]["A"])

            return Z, G, Q, T, A


if __name__ == "__main__":
    pass

