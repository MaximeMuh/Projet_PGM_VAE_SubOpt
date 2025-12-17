### On reprend les distributions du papier et on ajoute ContextualFlow.
### On nettoie et commente le code.
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import lognormal2 as lognormal, lognormal333

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


class Gaussian(nn.Module):
    def __init__(self, hyper_config):
        super(Gaussian, self).__init__()
        self.z_size, self.x_size = hyper_config["z_size"], hyper_config["x_size"]
        self.device = device

    def sample(self, mean, logvar, k):
        # On échantillonne z = mu + sigma * eps, avec eps ~ N(0, I)
        B = mean.size(0)
        eps = torch.randn(k, B, self.z_size, device=self.device)
        z = eps * torch.exp(0.5 * logvar) + mean          # [K,B,Z]
        logqz = lognormal(z, mean, logvar)                # log q(z|x) sous la base gaussienne
        return z, logqz

    def logprob(self, z, mean, logvar):
        # On évalue log N(z ; mean, diag(exp(logvar)))
        return lognormal(z, mean, logvar)


class Flow(nn.Module):
    def __init__(self, hyper_config):
        super(Flow, self).__init__()
        self.hyper_config = hyper_config
        self.z_size, self.x_size = hyper_config["z_size"], hyper_config["x_size"]
        self.act_func, self.device = hyper_config["act_func"], device

        # On construit r(v_T | z_T) : un MLP qui sort [mean_rv, logvar_rv]
        rv_arch = [[self.z_size, 50], [50, 50], [50, 2 * self.z_size]]
        self.rv_weights = nn.ModuleList([nn.Linear(a, b) for a, b in rv_arch])

        # On stocke les paramètres des flows (2 blocs, chaque bloc = 2 petites transforms)
        self.n_flows, h_s = 2, 50
        self.flow_params = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)]),
                nn.ModuleList([nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)])
            ]) for _ in range(self.n_flows)
        ])

    def norm_flow(self, params, z, v):
        # On fait une transformation affine sur v (conditionnée par z), puis sur z (conditionnée par v)
        h = torch.tanh(params[0][0](z))
        mew_, sig_ = params[0][1](h), torch.sigmoid(params[0][2](h))
        v = v * sig_ + mew_
        logdet = torch.sum(torch.log(sig_), dim=1)

        h = torch.tanh(params[1][0](v))
        mew_, sig_ = params[1][1](h), torch.sigmoid(params[1][2](h))
        z = z * sig_ + mew_
        logdet += torch.sum(torch.log(sig_), dim=1)

        return z, v, logdet

    def sample(self, mean, logvar, k):
        # On sample (z0, v0) gaussiens, puis on pousse (z, v) à travers les flows
        B, gaus = mean.size(0), Gaussian(self.hyper_config)

        z, logqz0 = gaus.sample(mean, logvar, k)                          # [K,B,Z]
        zeros = torch.zeros(B, self.z_size, device=self.device)
        v, logqv0 = gaus.sample(zeros, zeros, k)                          # [K,B,Z]

        z, v = z.view(-1, self.z_size), v.view(-1, self.z_size)           # [K*B,Z]
        logdetsum = torch.zeros(z.size(0), device=self.device)

        for flow_block in self.flow_params:
            z, v, logdet = self.norm_flow(flow_block, z, v)
            logdetsum += logdet

        logdetsum = logdetsum.view(k, B)

        # On calcule les paramètres de r(v_T|z_T) avec le petit MLP rv
        out = z
        for layer in self.rv_weights[:-1]:
            out = self.act_func(layer(out))
        out = self.rv_weights[-1](out)

        mean_rv, logvar_rv = out[:, :self.z_size], out[:, self.z_size:]
        mean_rv, logvar_rv = mean_rv.view(k, B, self.z_size), logvar_rv.view(k, B, self.z_size)

        # On évalue log r(v_T|z_T) et on combine tout dans log q(z_T|x)
        v = v.view(k, B, self.z_size)
        logrvT = lognormal333(v, mean_rv, logvar_rv)

        logpz = logqz0 + logqv0 - logdetsum - logrvT
        return z.view(k, B, self.z_size), logpz



class ContextualFlow(nn.Module):
    """
    Flux conditionnel : les transformations dépendent d'un contexte c (sorti par l'encodeur).
    Idée simple : au lieu d'avoir des couplings fixes, on donne aussi c au petit réseau
    pour que (scale, shift) dépendent de l'entrée.
    """
    def __init__(self, hyper_config):
        super(ContextualFlow, self).__init__()
        self.hyper_config = hyper_config
        self.z_size, self.x_size = hyper_config["z_size"], hyper_config["x_size"]
        self.act_func, self.device = hyper_config["act_func"], device

        # On lit les hyperparams, avec les mêmes valeurs par défaut
        self.context_size = hyper_config.get("context_size", 128)
        self.n_flows = hyper_config.get("n_flows", 4)
        self.z_half_size = self.z_size // 2
        h_s = hyper_config.get("h_s", 100)

        # Chaque bloc produit (t, s) en prenant [z_half ; context]
        self.flow_params = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([nn.Linear(self.z_half_size + self.context_size, h_s),
                               nn.Linear(h_s, self.z_half_size),
                               nn.Linear(h_s, self.z_half_size)]),
                nn.ModuleList([nn.Linear(self.z_half_size + self.context_size, h_s),
                               nn.Linear(h_s, self.z_half_size),
                               nn.Linear(h_s, self.z_half_size)])
            ]) for _ in range(self.n_flows)
        ])

    def norm_flow(self, params, z1, z2, context_k):
        # Étape 1 : on transforme z2 avec (z1, context)
        z1c = torch.cat([z1, context_k], dim=1)
        h = torch.tanh(params[0][0](z1c))
        mew_, sig_ = params[0][1](h), torch.sigmoid(params[0][2](h) + 2.0)
        z2 = z2 * sig_ + mew_
        logdet = torch.sum(torch.log(sig_), dim=1)

        # Étape 2 : on transforme z1 avec (z2, context) (z2 a déjà bougé)
        z2c = torch.cat([z2, context_k], dim=1)
        h = torch.tanh(params[1][0](z2c))
        mew_, sig_ = params[1][1](h), torch.sigmoid(params[1][2](h) + 2.0)
        z1 = z1 * sig_ + mew_
        logdet += torch.sum(torch.log(sig_), dim=1)

        return z1, z2, logdet

    def sample(self, mean, logvar, context, k):
        # On sample z0, puis on duplique le contexte k fois pour matcher [K*B, ...]
        B, gaus = mean.size(0), Gaussian(self.hyper_config)
        z, logqz0 = gaus.sample(mean, logvar, k)          # [K,B,Z]
        z = z.view(-1, self.z_size)                       # [K*B,Z]

        context_k = (context.unsqueeze(0)
                           .expand(k, B, self.context_size)
                           .contiguous()
                           .view(-1, self.context_size))  # [K*B,C]

        # On split z, on applique les flows, puis on recombine
        z1, z2 = z[:, :self.z_half_size], z[:, self.z_half_size:]
        logdetsum = torch.zeros(z.size(0), device=self.device)

        for flow_block in self.flow_params:
            z1, z2, logdet = self.norm_flow(flow_block, z1, z2, context_k)
            logdetsum += logdet

        logdetsum = logdetsum.view(k, B)
        z = torch.cat([z1, z2], dim=1).view(k, B, self.z_size)
        
        logpz = logqz0 - logdetsum
        return z, logpz