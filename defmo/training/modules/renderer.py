import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck
from .utils import group_norm


class Renderer(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = self.models(model, **kwargs)

    def forward(self, latent, n_frames):
        b, _, w, h = latent.shape
        ts = torch.linspace(0, 1, n_frames, device=latent.device)
        inputs = [torch.cat((t.repeat(b, 1, w, h), latent), 1) for t in ts]
        renders = torch.stack([self.model(i) for i in inputs], 1)
        return renders

    def models(_, name, **kwargs):
        def resnet_gn():
            return nn.Sequential(
                nn.Conv2d(2049, 1024, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, 1024),
                nn.ReLU(inplace=True),
                Bottleneck(1024, 256, norm_layer=group_norm()),
                nn.PixelShuffle(2),
                Bottleneck(256, 64, norm_layer=group_norm(8)),
                nn.PixelShuffle(2),
                Bottleneck(64, 16, norm_layer=group_norm(4)),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(4, 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def resnet_gn_tsweight(scale=1, shift=0):
            model = resnet_gn()
            with torch.no_grad():
                model[0].weight[:, 0] *= scale
                model[0].weight[:, 0] += shift
            return model

        resnet_gn_tsweight_3_0 = lambda: resnet_gn_tsweight(3, 0)
        resnet_gn_tsweight_0_08 = lambda: resnet_gn_tsweight(0, 0.08)
        resnet_gn_tsweight_3_08 = lambda: resnet_gn_tsweight(3, 0.08)

        try:
            return locals()[name](**kwargs)
        except KeyError:
            raise ValueError(f"Renderer model '{name}' not found.")
