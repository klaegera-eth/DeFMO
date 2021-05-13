import torch
import torch.nn as nn

from torchvision.models.resnet import Bottleneck
from .utils import group_norm


class Renderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = self.models(model)

    def forward(self, latent, n_frames):
        b, _, w, h = latent.shape
        ts = torch.linspace(0, 1, n_frames).to(latent.device)
        inputs = [torch.cat((t.repeat(b, 1, w, h), latent), 1) for t in ts]
        renders = torch.stack([self.model(i) for i in inputs], 1)
        return renders

    def models(_, name):
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

        def resnet():
            return nn.Sequential(
                nn.Conv2d(2049, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(
                    1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
                Bottleneck(1024, 256),
                nn.PixelShuffle(2),
                Bottleneck(256, 64),
                nn.PixelShuffle(2),
                Bottleneck(64, 16),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid(),
            )

        def resnet_smaller():
            return nn.Sequential(
                nn.Conv2d(1025, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                Bottleneck(1024, 256),
                nn.PixelShuffle(2),
                Bottleneck(256, 64),
                nn.PixelShuffle(2),
                Bottleneck(64, 16),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid(),
            )

        def cnn():
            return nn.Sequential(
                nn.Conv2d(513, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(
                    1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(
                    256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(
                    16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
                ),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid(),
            )

        return locals()[name]()
