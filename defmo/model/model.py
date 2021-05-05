import torch
import torch.nn as nn

from .encoder import Encoder
from .renderer import Renderer
from .loss import Loss


class Model(nn.Module):
    def __init__(self, encoder, renderer, losses=[]):
        super().__init__()
        self.models = {"encoder": encoder, "renderer": renderer}
        self.encoder = Encoder(encoder)
        self.renderer = Renderer(renderer)
        self.loss = Loss(losses)

    def forward(self, inputs, apply_loss=True):
        imgs = inputs["imgs"][:, 1], inputs["imgs"][:, 2]
        latent = self.encoder(torch.cat(imgs, 1))
        renders = self.renderer(latent, n_frames=inputs["frames"].shape[1])
        outputs = dict(latent=latent, renders=renders)
        if apply_loss:
            outputs = self.loss(inputs, outputs)
        return outputs

    def state_dict(self, *args, **kwargs):
        return {"models": self.models, "state": super().state_dict(*args, **kwargs)}

    def load_state_dict(self, state_dict, strict=True):
        self.models = state_dict["models"]
        self.encoder = Encoder(self.models["encoder"])
        self.renderer = Renderer(self.models["renderer"])
        super().load_state_dict(state_dict["state"], strict)
