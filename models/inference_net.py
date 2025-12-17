### On s'inspire du code du papier qu'on améliore et adapte notamment pour ContextualFlow.
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

sys.path.insert(0, "utils")
from utils import lognormal2 as lognormal
from utils import lognormal333

from distributions import Gaussian, Flow


class standard(nn.Module):
    def __init__(self, hyper_config):
        super().__init__()

        if torch.backends.mps.is_available():
            self.dtype, self.device = torch.FloatTensor, torch.device("mps")
        elif torch.cuda.is_available():
            self.dtype, self.device = torch.cuda.FloatTensor, torch.device("cuda")
        else:
            self.dtype, self.device = torch.FloatTensor, torch.device("cpu")

        # On stocke les hyperparams utiles
        self.hyper_config = hyper_config
        self.z_size, self.x_size = hyper_config["z_size"], hyper_config["x_size"]
        self.act_func = hyper_config["act_func"]

        # On garde context_size
        self.context_size = hyper_config.get("context_size", 0)

        # On construit l’encodeur 
        self.encoder_weights = nn.ModuleList(
            [nn.Linear(a, b) for (a, b) in hyper_config["encoder_arch"]]
        )

        # On garde q tel quel (Gaussian / Flow / HNF / ContextualFlow…)
        self.q = hyper_config["q"]

        # On déplace tout le module sur le device
        self.to(self.device)

    def forward(self, k, x, logposterior):
        """
        k : nb de samples
        x : [B, X]
        logposterior(z) -> [P, B] (sert seulement si hnf=True)
        """
        self.B = x.size(0)

        # On encode x avec le MLP
        out = x
        for layer in self.encoder_weights[:-1]:
            out = self.act_func(layer(out))
        out = self.encoder_weights[-1](out)

        # On découpe la sortie : mean | logvar | (optionnel) context
        mean   = out[:, :self.z_size]                           # [B, Z]
        logvar = out[:, self.z_size:2 * self.z_size]            # [B, Z]

        # On choisit comment on sample z suivant la config, sans changer la logique
        if self.hyper_config.get("hnf", False):
            # Cas HNF : on passe logposterior en plus
            z, logqz = self.q.sample(mean, logvar, k, logposterior)

        elif self.context_size > 0:
            # Cas flow conditionné : on récupère le contexte après mean+logvar
            context = out[:, 2 * self.z_size:]
            z, logqz = self.q.sample(mean, logvar, context=context, k=k)

        else:
            # Cas standard : Gaussian / Flow / Flow1
            z, logqz = self.q.sample(mean, logvar, k=k)

        return z, logqz