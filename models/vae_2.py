### On reprend en grande partie le code du papier, car le VAE utilisé est un modèle classique et ne constitue pas en soi une contribution nouvelle.
### Le code est ensuite nettoyé, clarifié et commenté, puis adapté et étendu, en particulier pour intégrer le Contextual Flow.
import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")
import time, sys, os

sys.path.insert(0, "utils")

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import log_bernoulli
from generator import Generator


class VAE(nn.Module):
    def __init__(self, hyper_config, seed=1):
        super(VAE, self).__init__()
        torch.manual_seed(seed)

        self.z_size, self.x_size = hyper_config["z_size"], hyper_config["x_size"]
        self.act_func = hyper_config["act_func"]

        # On construit la q(z|x) choisie dans la config (Gaussian / Flow / HNF / ContextualFlow, etc.)
        self.q_dist = hyper_config["q_dist"](hyper_config=hyper_config)

        # On sépare le décodeur dans une classe Generator
        self.generator = Generator(hyper_config=hyper_config)

        if torch.backends.mps.is_available():
            self.dtype, self.device = torch.FloatTensor, torch.device("mps")
            self.q_dist.to(self.device); self.generator.to(self.device)
        elif torch.cuda.is_available():
            self.dtype, self.device = torch.cuda.FloatTensor, torch.device("cuda")
            self.q_dist.cuda(); self.generator.cuda()
        else:
            self.dtype, self.device = torch.FloatTensor, torch.device("cpu")

    def _make_zeros(self, B):
        if torch.backends.mps.is_available():
            return Variable(torch.zeros(B, self.z_size).to(torch.device("mps")))
        if torch.cuda.is_available():
            return Variable(torch.zeros(B, self.z_size).type(self.dtype))
        return Variable(torch.zeros(B, self.z_size))

    def forward(self, x, k, warmup=1.0):
        # On prépare les tailles batch et le prior p(z)=N(0,I)
        self.B = x.size()[0]
        self.zeros = self._make_zeros(self.B)

        # On définit log p(z) + log p(x|z), avec Bernoulli sur les logits du décodeur
        self.logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.generator.decode(aa), x)

        # On échantillonne z ~ q(z|x) et on récupère log q(z|x)
        z, logqz = self.q_dist.forward(k, x, self.logposterior)
        logpxz = self.logposterior(z)

        # On calcule l'objectif : log p(x,z) - warmup * log q(z|x)
        elbo = logpxz - (warmup * logqz)  # [K,B]

        # Si k>1, on passe en IWAE (log-mean-exp sur K)
        if k > 1:
            max_ = torch.max(elbo, 0)[0]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_

        # On finit comme avant : moyenne batch + scalaires
        elbo = torch.mean(elbo)
        logpxz = torch.mean(logpxz)
        logqz = torch.mean(logqz)
        return elbo, logpxz, logqz

    def sample_q(self, x, k):
        # On veut juste sortir des z ~ q(z|x)
        self.B = x.size()[0]
        self.zeros = self._make_zeros(self.B)

        self.logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.generator.decode(aa), x)
        z, logqz = self.q_dist.forward(k=k, x=x, logposterior=self.logposterior)
        return z

    def logposterior_func(self, x, z):
        self.B = x.size()[0]
        self.zeros = self._make_zeros(self.B)

        z = Variable(z).type(self.dtype)
        z = z.view(-1, self.B, self.z_size)
        return lognormal(z, self.zeros, self.zeros) + log_bernoulli(self.generator.decode(z), x)

    def logposterior_func2(self, x, z):
        self.B = x.size()[0]
        self.zeros = self._make_zeros(self.B)

        z = z.view(-1, self.B, self.z_size)
        return lognormal(z, self.zeros, self.zeros) + log_bernoulli(self.generator.decode(z), x)

    def forward2(self, x, k):
        self.B = x.size()[0]
        if torch.backends.mps.is_available():
            zeros = torch.zeros(self.B, self.z_size, device="mps")
        elif torch.cuda.is_available():
            zeros = torch.zeros(self.B, self.z_size, device="cuda")
        else:
            zeros = torch.zeros(self.B, self.z_size)

        self.logposterior = lambda z: lognormal(z, zeros, zeros) + log_bernoulli(self.generator.decode(z), x)

        z, logqz = self.q_dist.forward(k, x, self.logposterior)
        logpxz = self.logposterior(z)

        log_w = logpxz - logqz
        max_logw, _ = torch.max(log_w, 0)
        iwae = torch.log(torch.mean(torch.exp(log_w - max_logw), 0)) + max_logw
        elbo = torch.mean(iwae)
        return elbo, torch.mean(logpxz), torch.mean(logqz)

    def decode(self, z):
        return self.generator.decode(z)

    def forward3_prior(self, x, k):
        # Version “prior only” : on sample z ~ N(0,I) et on regarde juste log p(x|z)
        self.B = x.size()[0]
        self.zeros = self._make_zeros(self.B)

        self.logposterior = lambda aa: log_bernoulli(self.generator.decode(aa), x)

        if torch.backends.mps.is_available():
            z = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().to(torch.device("mps")))
        elif torch.cuda.is_available():
            z = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype))
        else:
            z = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_())

        logpxz = self.logposterior(z)
        elbo = logpxz
        if k > 1:
            max_ = torch.max(elbo, 0)[0]
            elbo = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_
        elbo = torch.mean(elbo)
        return elbo

    def forward3(self, x, k, warmup=1.0, return_components=False):
        # Forward standard avec option de renvoyer recon / kl, mêmes formules
        self.B = x.size()[0]
        if torch.backends.mps.is_available():
            self.zeros = torch.zeros(self.B, self.z_size, device="mps")
        elif torch.cuda.is_available():
            self.zeros = torch.zeros(self.B, self.z_size, device="cuda")
        else:
            self.zeros = torch.zeros(self.B, self.z_size)

        self.logposterior = lambda aa: lognormal(aa, self.zeros, self.zeros) + log_bernoulli(self.generator.decode(aa), x)

        z, logqz = self.q_dist.forward(k, x, self.logposterior)
        logpxz = self.logposterior(z)

        # On calcule recon et KL exactement comme dans la version donnée
        kl = (logqz - lognormal(z, self.zeros, self.zeros)).mean()
        recon = log_bernoulli(self.generator.decode(z), x).mean()
        elbo = recon - warmup * kl

        # IWAE si k>1, même expression (logpxz - warmup*logqz)
        if k > 1:
            log_w = logpxz - warmup * logqz
            max_logw, _ = torch.max(log_w, 0)
            elbo = torch.log(torch.mean(torch.exp(log_w - max_logw), 0)) + max_logw
            elbo = torch.mean(elbo)

        return (elbo, recon, kl) if return_components else (elbo, logpxz.mean(), logqz.mean())