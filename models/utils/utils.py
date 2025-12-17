### On reprend les fonctions utilitaires du papier, on les nettoie et adapte un peu notamment à l'environnement MPS.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# On pose une version simple de LayerNorm (mean/std sur la dernière dim)
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta  = nn.Parameter(torch.zeros(features))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def lognormal2(x, mean, logvar):
    """
    On calcule log N(x ; mean, diag(exp(logvar))).
    Ici x est en [P,B,Z], mean/logvar en [B,Z], et on renvoie [P,B].
    """
    assert len(x.size()) == 3 and len(mean.size()) == 2 and len(logvar.size()) == 2
    assert x.size(1) == mean.size(0)

    D = x.size(2)

    # On garde exactement la même logique de term1 
    if torch.backends.mps.is_available():
        term1 = D * torch.log(torch.FloatTensor([2. * math.pi])).to(x.device)
    elif torch.cuda.is_available():
        term1 = D * torch.log(torch.cuda.FloatTensor([2. * math.pi]))
    else:
        term1 = D * torch.log(torch.FloatTensor([2. * math.pi]))

    return -0.5 * (Variable(term1) + logvar.sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(2))


def lognormal333(x, mean, logvar):
    """
    Même idée, mais mean/logvar dépendent aussi de P : mean/logvar en [P,B,Z].
    On renvoie [P,B].
    """
    assert len(x.size()) == 3 and len(mean.size()) == 3 and len(logvar.size()) == 3
    assert x.size(0) == mean.size(0) and x.size(1) == mean.size(1)

    D = x.size(2)

    if torch.backends.mps.is_available():
        term1 = D * torch.log(torch.FloatTensor([2. * math.pi])).to(x.device)
    elif torch.cuda.is_available():
        term1 = D * torch.log(torch.cuda.FloatTensor([2. * math.pi]))
    else:
        term1 = D * torch.log(torch.FloatTensor([2. * math.pi]))

    return -0.5 * (Variable(term1) + logvar.sum(2) + ((x - mean).pow(2) / torch.exp(logvar)).sum(2))


def log_bernoulli(pred_no_sig, target):
    """
    On calcule log p(target | logits=pred_no_sig) en mode stable numériquement.
    pred_no_sig : [P,B,X], target : [B,X], sortie : [P,B].
    """
    assert len(pred_no_sig.size()) == 3 and len(target.size()) == 2
    assert pred_no_sig.size(1) == target.size(0)

    # Formule stable du BCE avec logits, et on somme sur X
    return -(torch.clamp(pred_no_sig, min=0)
             - pred_no_sig * target
             + torch.log(1. + torch.exp(-torch.abs(pred_no_sig)))).sum(2)


def lognormal3(x, mean, logvar):
    """
    Version 1D : x/mean/logvar en [P], on renvoie un scalaire
    """
    return -0.5 * (logvar.sum(0) + ((x - mean).pow(2) / torch.exp(logvar)).sum(0))


def lognormal4(x, mean, logvar):
    """
    Version batch : x en [B,X], mean/logvar en [X], sortie en [B].
    """
    D = x.size(1)
    term1 = D * torch.log(torch.FloatTensor([2. * math.pi]))
    aaa = -0.5 * (term1 + logvar.sum(0) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))
    return aaa