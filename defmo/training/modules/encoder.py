import torch.nn as nn

from torchvision.models import resnet50


class Encoder(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = self.models(model, **kwargs)

    def forward(self, inputs):
        return self.model(inputs)

    def models(_, name, **kwargs):
        def _resnet_norm(norm_layer):
            model = resnet50(norm_layer=norm_layer)
            model.load_state_dict(
                resnet50(pretrained=True).state_dict(),
                strict=False,
            )
            return model

        def _resnet_nomaxpool(resnet):
            conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            conv.load_state_dict({"weight": resnet.conv1.weight.repeat(1, 2, 1, 1)})

            return nn.Sequential(
                Encoder.ImgsToChannels(),
                conv,
                resnet.bn1,
                resnet.relu,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )

        def resnet_gn_nomaxpool():
            from .utils import group_norm

            resnet = _resnet_norm(group_norm())
            return _resnet_nomaxpool(resnet)

        def v2():
            resnet = resnet50(pretrained=True)
            return _resnet_nomaxpool(resnet)

        try:
            return locals()[name](**kwargs)
        except KeyError:
            raise ValueError(f"Encoder model '{name}' not found.")

    class ImgsToChannels(nn.Module):
        def forward(self, imgs):
            bs, _, _, *yx = imgs.shape
            return imgs.reshape(bs, -1, *yx)
