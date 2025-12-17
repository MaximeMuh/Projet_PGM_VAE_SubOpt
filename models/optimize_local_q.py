### On s'inspire du code du papier qu'on améliore et adapte notamment pour ContextualFlow.
import numpy as np
import time
import pickle

import torch
from torch.autograd import Variable
import torch.optim as optim

from utils import lognormal2 as lognormal
from utils import lognormal333
from utils import log_bernoulli

quick = 0


def optimize_local_q_dist(logposterior, hyper_config, x, q):
    """
    On optimise un q(z|x) local (mean, logvar, et parfois context).
    On garde exactement les 3 cas :
      - HNF : q.sample(mean, logvar, P, logposterior)
      - ContextualFlow : q.sample(mean, logvar, context=context, k=P)
      - Standard : q.sample(mean, logvar, k=P)
    On renvoie VAE (= ELBO moyen) et IWAE (log-mean-exp).
    """

    B = x.size()[0]      # batch size
    P = 50               # samples pendant l'optim
    P_final = 5000       # samples pour l'éval finale

    z_size = hyper_config['z_size']
    context_size = hyper_config.get('context_size', 0)
    is_hnf = hyper_config.get('hnf', False)

    if torch.backends.mps.is_available(): device_tensor = torch.device("mps")
    elif torch.cuda.is_available():       device_tensor = torch.device("cuda")
    else:                                 device_tensor = torch.device("cpu")

    # On crée les params variationnels locaux 
    mean   = Variable(torch.zeros(B, z_size).to(device_tensor), requires_grad=True)
    logvar = Variable(torch.zeros(B, z_size).to(device_tensor), requires_grad=True)
    params = [mean, logvar]

    # On ajoute un context si on est en ContextualFlow
    context = None
    if context_size > 0:
        context = Variable(torch.zeros(B, context_size).to(device_tensor), requires_grad=True)
        params.append(context)

    # On ajoute les paramètres du flow q
    for a in q.parameters(): params.append(a)

    optimizer = optim.Adam(params, lr=.001)

    loss_window, best_window_avg, n_bad_windows = [], -1, 0

    for epoch in range(1, 999999):
        # On sample z selon le bon cas 
        if is_hnf:
            z, logqz = q.sample(mean, logvar, P, logposterior)
        elif context is not None:
            z, logqz = q.sample(mean, logvar, context=context, k=P)
        else:
            z, logqz = q.sample(mean, logvar, k=P)

        # On calcule log p(x,z) via logposterior, puis la loss = -E[logp - logq]
        logpx = logposterior(z)
        optimizer.zero_grad()
        loss = -(torch.mean(logpx - logqz))
        loss_np = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        loss_window.append(loss_np)
        if epoch % 100 == 0:
            window_avg_loss = np.mean(loss_window)
            if window_avg_loss < best_window_avg or best_window_avg == -1:
                n_bad_windows = 0
                best_window_avg = window_avg_loss
            else:
                n_bad_windows += 1
                if n_bad_windows > 10: break

            if epoch % 2000 == 0:
                print(f"Epoch {epoch}, Avg Loss: {window_avg_loss:.4f}, Worse: {n_bad_windows}")

            loss_window = []

    if is_hnf:
        z, logqz = q.sample(mean, logvar, P_final, logposterior)
    elif context is not None:
        z, logqz = q.sample(mean, logvar, context=context, k=P_final)
    else:
        z, logqz = q.sample(mean, logvar, k=P_final)

    # On calcule ELBO / IWAE 
    logpx = logposterior(z)
    elbo = logpx - logqz  # [P,B]

    vae = torch.mean(elbo)

    max_ = torch.max(elbo, 0)[0]
    elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_  # [B]
    iwae = torch.mean(elbo_)

    return vae, iwae