import torch.nn as nn

from torchvision.models import resnet50

from .utils import group_norm


class Encoder(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = self.models(model, **kwargs)

    def forward(self, inputs):
        return self.model(inputs)

    def models(_, name, **kwargs):
        def resnet_gn_nomaxpool():
            resnet = Encoder.resnet_with_norm(group_norm())
            return Encoder.resnet_double_input(resnet, maxpool=False)

        def resnet_gn():
            resnet = Encoder.resnet_with_norm(group_norm())
            return Encoder.resnet_double_input(resnet)

        def resnet_gn_single_nomaxpool():
            resnet = Encoder.resnet_with_norm(group_norm())
            return Encoder.resnet_single(resnet, maxpool=False)

        def resnet_gn_single():
            resnet = Encoder.resnet_with_norm(group_norm())
            return Encoder.resnet_single(resnet)

        def v2():
            resnet = resnet50(pretrained=True)
            return Encoder.resnet_double_input(resnet, maxpool=False)

        try:
            return locals()[name](**kwargs)
        except KeyError:
            raise ValueError(f"Encoder model '{name}' not found.")

    @staticmethod
    def resnet_with_norm(norm_layer):
        model = resnet50(norm_layer=norm_layer)
        model.load_state_dict(
            resnet50(pretrained=True).state_dict(),
            strict=False,
        )
        return model

    @staticmethod
    def resnet_double_input(resnet, maxpool=True):
        conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv.load_state_dict({"weight": resnet.conv1.weight.repeat(1, 2, 1, 1)})

        return nn.Sequential(
            Encoder.ImgsToChannels(),
            conv,
            resnet.bn1,
            resnet.relu,
            *[resnet.maxpool] if maxpool else [],
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    @staticmethod
    def resnet_single(resnet, maxpool=True):
        return nn.Sequential(
            Encoder.ImgsToChannels(),
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            *[resnet.maxpool] if maxpool else [],
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

    class ImgsToChannels(nn.Module):
        def forward(self, imgs):
            bs, _, _, *yx = imgs.shape
            return imgs.reshape(bs, -1, *yx)
