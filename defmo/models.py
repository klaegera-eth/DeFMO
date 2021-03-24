import torch
import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import Bottleneck


class Encoder(nn.Module):
    def __init__(self, model="v2"):
        super().__init__()
        self.model = self.models(model)

    def forward(self, inputs):
        return self.model(inputs)

    def models(_, name):
        def v1():
            model = torchvision.models.resnet50(pretrained=True)
            modelc = nn.Sequential(*list(model.children())[:-2])
            pretrained_weights = modelc[0].weight
            modelc[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            modelc[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
            modelc[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
            return nn.Sequential(modelc, nn.PixelShuffle(2))

        def v2():
            model = torchvision.models.resnet50(pretrained=True)
            modelc1 = nn.Sequential(*list(model.children())[:3])
            modelc2 = nn.Sequential(*list(model.children())[4:8])
            pretrained_weights = modelc1[0].weight
            modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
            modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
            modelc = nn.Sequential(modelc1, modelc2)
            return modelc

        def v3():
            model = torchvision.models.resnet50(pretrained=True)
            modelc1 = nn.Sequential(*list(model.children())[:3])
            modelc2 = nn.Sequential(*list(model.children())[4:8])
            pretrained_weights = modelc1[0].weight
            modelc1[0] = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
            modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
            modelc3 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, bias=False)
            modelc = nn.Sequential(modelc1, modelc2, modelc3)
            return modelc

        return locals()[name]()


class Renderer(nn.Module):
    def __init__(self, n_frames, model="resnet"):
        super().__init__()
        self.ts = torch.linspace(0, 1, n_frames)
        self.model = self.models(model)

    def forward(self, latent):
        b, _, w, h = latent.shape
        inputs = [torch.cat((ts.repeat(b, 1, w, h), latent), 1) for ts in self.ts]
        renders = torch.stack([self.model(i) for i in inputs], 1)
        return nn.Sigmoid(renders)

    def models(_, name):
        def resnet():
            return nn.Sequential(
                nn.Conv2d(2049, 1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
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
                nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid(),
            )

        return locals()[name]()
