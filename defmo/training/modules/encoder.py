import torch.nn as nn

from torchvision.models import resnet50


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = self.models(model)

    def forward(self, inputs):
        return self.model(inputs)

    def models(_, name):
        def _resnet_gn():
            from .utils import group_norm

            model = resnet50(norm_layer=group_norm())
            model.load_state_dict(
                resnet50(pretrained=True).state_dict(),
                strict=False,
            )
            return model

        def resnet_gn_nomaxpool():
            resnet = _resnet_gn()

            conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            conv.load_state_dict({"weight": resnet.conv1.weight.repeat(1, 2, 1, 1)})

            return nn.Sequential(
                conv,
                resnet.bn1,
                resnet.relu,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )

        def v1():
            model = resnet50(pretrained=True)
            modelc = nn.Sequential(*list(model.children())[:-2])
            pretrained_weights = modelc[0].weight
            modelc[0] = nn.Conv2d(
                6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            modelc[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
            modelc[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
            return nn.Sequential(modelc, nn.PixelShuffle(2))

        def v2():
            model = resnet50(pretrained=True)
            modelc1 = nn.Sequential(*list(model.children())[:3])
            modelc2 = nn.Sequential(*list(model.children())[4:8])
            pretrained_weights = modelc1[0].weight
            modelc1[0] = nn.Conv2d(
                6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
            modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
            modelc = nn.Sequential(modelc1, modelc2)
            return modelc

        def v3():
            model = resnet50(pretrained=True)
            modelc1 = nn.Sequential(*list(model.children())[:3])
            modelc2 = nn.Sequential(*list(model.children())[4:8])
            pretrained_weights = modelc1[0].weight
            modelc1[0] = nn.Conv2d(
                6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            modelc1[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
            modelc1[0].weight.data[:, 3:, :, :] = nn.Parameter(pretrained_weights)
            modelc3 = nn.Conv2d(
                2048, 1024, kernel_size=3, stride=1, padding=1, bias=False
            )
            modelc = nn.Sequential(modelc1, modelc2, modelc3)
            return modelc

        return locals()[name]()
