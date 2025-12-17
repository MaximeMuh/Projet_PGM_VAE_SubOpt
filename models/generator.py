### On s'inspire du code du papier qu'on nettoie, commente et améliore.
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hyper_config):
        super().__init__()

        if torch.backends.mps.is_available():
            self.dtype, self.device = torch.FloatTensor, torch.device("mps")
            self.to(self.device)
        elif torch.cuda.is_available() and hyper_config.get("cuda", False):
            self.dtype, self.device = torch.cuda.FloatTensor, torch.device("cuda")
            self.cuda()
        else:
            self.dtype, self.device = torch.FloatTensor, torch.device("cpu")

        # On récupère les tailles et l'activation
        self.z_size = hyper_config["z_size"]
        self.x_size = hyper_config["x_size"]
        self.act_func = hyper_config["act_func"]

        # On construit le décodeur
        self.decoder_weights = nn.ModuleList(
            [nn.Linear(a, b) for (a, b) in hyper_config["decoder_arch"]]
        )

    def decode(self, z):
        # z est en (k, B, z_size)
        k, B = z.size(0), z.size(1)
        out = z.view(-1, self.z_size)  # (k*B, z_size)

        # On passe couche par couche, activation partout sauf la dernière
        for layer in self.decoder_weights[:-1]:
            out = self.act_func(layer(out))
        out = self.decoder_weights[-1](out)

        # On remet en (k, B, x_size)
        return out.view(k, B, self.x_size)