import torch
import torch.nn as nn

from .encoder import Encoder
from .renderer import Renderer
from .loss import Loss


class Model(nn.Module):
    def __init__(self, losses=[], encoder=None, renderer=None, checkpoint=None):
        super().__init__()
        self.models = dict(encoder=encoder, renderer=renderer)
        if checkpoint is not None:
            checkpoint = checkpoint["model"]
            self.models = checkpoint["models"]
        self.encoder = Encoder(self.models["encoder"])
        self.renderer = Renderer(self.models["renderer"])
        self.loss = Loss(losses)
        if checkpoint is not None:
            self.load_state_dict(checkpoint["state"])

    def forward(self, inputs, apply_loss=True):
        imgs = inputs["imgs"][:, 1], inputs["imgs"][:, 2]
        latent = self.encoder(torch.cat(imgs, 1))
        renders = self.renderer(latent, n_frames=inputs["frames"].shape[1])
        outputs = dict(latent=latent, renders=renders)
        if apply_loss:
            outputs = self.loss(inputs, outputs)
        return outputs

    def get_state(self):
        return dict(models=self.models, state=self.state_dict())
