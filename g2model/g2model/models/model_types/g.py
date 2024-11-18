import torch
import pyro
from pyro.ops.indexing import Vindex
import pyro.distributions as dist
from pyro.distributions import transforms


class Model():

    def __init__(self, n_c):
        self.n_c = n_c
        self.model_type = "model_g"
        print(f"{self.model_type} n_c: {n_c}")

        if torch.device("cuda" if torch.cuda.is_available() else "cpu").type == "cuda":
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def model(self, data=None):

        ## Prior_____________

        #base_G_c = dist.Normal(torch.ones(self.n_c) * (4 / self.n_c), torch.ones(self.n_c))
        base_G_c = dist.Normal(torch.ones(self.n_c) * (-1.0), torch.ones(self.n_c) * (1.0))
        #lam_G_c = pyro.sample("lam_G_c", dist.TransformedDistribution(base_G_c, [transforms.OrderedTransform()]))
        #std_G_c = pyro.sample("std_G_c", dist.Gamma(torch.ones(self.n_c), torch.ones(self.n_c)).to_event(1))
        mean_G_c = pyro.sample("mean_G_c", dist.TransformedDistribution(base_G_c, [transforms.OrderedTransform(),transforms.SigmoidTransform()]))
        conc_G_c = pyro.sample("conc_G_c", dist.Gamma(torch.ones(self.n_c), torch.ones(self.n_c)).to_event(1))

        pi_Z_c = pyro.sample("pi_Z_c", dist.Dirichlet(torch.ones(self.n_c) / self.n_c))  ## translates into number of classes

        ## Likelihood ______________________________________

        with pyro.plate('data_plate', data["mask"]["G"].shape[0]):

            Z = pyro.sample('Z', dist.Categorical(pi_Z_c), infer={"enumerate": "parallel"})

            mean_G_c = Vindex(mean_G_c)[..., Z.long()]
            conc_G_c = Vindex(conc_G_c)[..., Z.long()] + 2.0

            #G = pyro.sample('G', dist.Normal(Vindex(lam_G_c)[..., Z.long()], Vindex(std_G_c)[..., Z.long()]).mask(data["mask"]["G"]), obs=data["data"]["G"])
            G = pyro.sample('G', dist.Beta(conc_G_c * mean_G_c, conc_G_c * (1 - mean_G_c)).mask(data["mask"]["G"]),obs=data["data"]["G"])
            return Z, G


if __name__ == "__main__":
    pass

