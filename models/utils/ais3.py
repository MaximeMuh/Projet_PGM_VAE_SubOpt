### On réécrit la fonction AIS pour estimer log p(x) avec AIS + HMC.

import math, time
import numpy as np
import torch

from utils import lognormal2 as lognormal
from utils import log_bernoulli


@torch.no_grad()
def _to_numpy(x):
    if isinstance(x, torch.Tensor): 
        return x.detach().cpu().numpy()
    return x


def test_ais(model, data_x, batch_size=100, display=10, k=10, n_intermediate_dists=500):
    """
    On estime log p(x) avec AIS + HMC.
    On fait un chemin p_t(z|x) ∝ p(z) p(x|z)^t avec t qui va de 0 à 1.
    À chaque pas on ajoute un incrément de poids puis on bouge z via HMC (avec MH).
    """
    dev = next(model.generator.parameters()).device
    dtype = next(model.generator.parameters()).dtype
    z_size, x_size = int(model.z_size), int(model.x_size)
    decode = model.decode if hasattr(model, "decode") else model.generator.decode

    if batch_size <= 0: raise ValueError("AIS error: batch_size must be > 0.")
    if n_intermediate_dists < 2: raise ValueError("AIS error: n_intermediate_dists must be >= 2.")
    if k <= 0: raise ValueError("AIS error: k must be > 0.")

    n_HMC_steps, step_size = 10, 0.1
    min_step, max_step = 1e-4, 0.5

    def intermediate_logprob(t_scalar, z, zeros, batch):
        # On calcule log p_t(z|x) = log p(z) + t * log p(x|z), résultat [k,B]
        log_prior = lognormal(z, zeros, zeros)
        x_logits  = decode(z)
        log_like  = log_bernoulli(x_logits, batch)
        return log_prior + float(t_scalar) * log_like

    def hmc_step(z, t_scalar, zeros, batch, step_size):
        # On fait un leapfrog multi-steps puis un MH, et on ajuste step_size un peu
        v = torch.randn_like(z)

        def U(z_): 
            return -intermediate_logprob(t_scalar, z_, zeros, batch)

        z0, v0 = z, v

        def K(v_):
            return 0.5 * (v_ * v_).view(v_.shape[0], v_.shape[1], -1).sum(dim=2)

        with torch.enable_grad():
            v = v - 0.5 * step_size * torch.autograd.grad(U(z).sum(), z, create_graph=False)[0]
            for _ in range(n_HMC_steps):
                z = z + step_size * v
                z.requires_grad_(True)
                v = v - step_size * torch.autograd.grad(U(z).sum(), z, create_graph=False)[0]
            v = v - 0.5 * step_size * torch.autograd.grad(U(z).sum(), z, create_graph=False)[0]

        H0, H1 = U(z0) + K(v0), U(z) + K(v)
        log_accept = -(H1 - H0)
        accept_prob = torch.exp(torch.clamp(log_accept, max=0.0))

        u = torch.rand_like(accept_prob)
        accept = (u < accept_prob).float().unsqueeze(-1)
        z = accept * z + (1 - accept) * z0

        acc_rate = accept_prob.mean().item()
        step_size = step_size * (1.02 if acc_rate > 0.65 else 0.98)
        step_size = float(min(max(step_size, min_step), max_step))

        z = z.detach()
        z.requires_grad_(True)
        return z, step_size

    X_np = _to_numpy(data_x)
    N = X_np.shape[0]
    n_batches = max(1, math.ceil(N / batch_size))

    schedule = torch.linspace(0.0, 1.0, steps=n_intermediate_dists, device=dev, dtype=dtype)

    t_start = time.time()
    batch_logws = []

    for b in range(n_batches):
        start, end = b * batch_size, min(N, (b + 1) * batch_size)
        B = end - start
        if B <= 0: 
            continue

        batch = torch.from_numpy(X_np[start:end]).to(dev, dtype)
        zeros = torch.zeros(B, z_size, device=dev, dtype=dtype)

        logw = torch.zeros(k, B, device=dev, dtype=dtype)
        z = torch.randn(k, B, z_size, device=dev, dtype=dtype, requires_grad=True)

        for t0_s, t1_s in zip(schedule[:-1], schedule[1:]):
            # On ajoute l’incrément de poids log p_{t1}(z) - log p_{t0}(z)
            with torch.no_grad():
                l0 = intermediate_logprob(t0_s.item(), z.detach(), zeros, batch)
                l1 = intermediate_logprob(t1_s.item(), z.detach(), zeros, batch)
                logw += (l1 - l0)

            # On fait une transition HMC sur la cible p_{t1}
            z, step_size = hmc_step(z, t1_s.item(), zeros, batch, step_size)

        max_ = torch.max(logw, dim=0).values
        log_mean_w = torch.log(torch.mean(torch.exp(logw - max_), dim=0)) + max_
        batch_logws.append(log_mean_w.mean().item())

        if display and int(display) > 0 and (b % int(display) == 0 or b == n_batches - 1):
            elapsed = time.time() - t_start
            print(f"[AIS] batch {b+1}/{n_batches}  mean(logw)={batch_logws[-1]:.3f}  "
                  f"step_size={step_size:.4f}  t={elapsed:.1f}s")

    mean_est = float(np.mean(batch_logws)) if len(batch_logws) else float("nan")
    print(f"[AIS] Final mean log p(x) ≈ {mean_est:.3f}   (over {len(batch_logws)} batches)")
    return mean_est