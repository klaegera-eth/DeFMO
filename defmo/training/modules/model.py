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
        latent = self.encoder(inputs["imgs"])
        renders = self.renderer(latent, n_frames=inputs["frames"].shape[1])
        outputs = dict(latent=latent, renders=renders)
        if apply_loss:
            outputs = self.loss(inputs, outputs)
        return outputs
