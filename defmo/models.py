import torch
import torch.nn as nn

from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck


class Model(nn.Module):
    def __init__(self, losses, encoder=None, renderer=None, checkpoint=None):
        super().__init__()
        self.models = dict(encoder=encoder, renderer=renderer)
        if checkpoint is not None:
            self.models = checkpoint["models"]
        self.encoder = Encoder(self.models["encoder"])
        self.renderer = Renderer(self.models["renderer"])
        self.loss = Loss(losses)
        if checkpoint is not None:
            self.load_state_dict(checkpoint["state"])

    def forward(self, inputs):
        outputs = self.process(inputs)
        return self.loss(inputs, outputs)

    def process(self, inputs):
        imgs = inputs["imgs"][:, 1], inputs["imgs"][:, 2]
        latent = self.encoder(torch.cat(imgs, 1))
        renders = self.renderer(latent, n_frames=inputs["frames"].shape[1])
        return dict(latent=latent, renders=renders)

    def get_state(self):
        return dict(models=self.models, state=self.state_dict())


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = self.models(model)

    def forward(self, inputs):
        return self.model(inputs)

    def models(_, name):
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


class Loss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = losses
        self.was_training = False

    def forward(self, inputs, outputs):
        losses = [loss(inputs, outputs) for loss in self.losses]
        return torch.stack(losses, 1)

    def record(self, batch):
        if self.training != self.was_training:
            self.was_training = self.training
            for loss in self.losses:
                loss.history.clear()
        for row in batch:
            for loss, val in zip(self.losses, row):
                loss.history.append(val.item())

    def mean(self, most_recent=None):
        if not self.losses:
            return 0
        means = [loss.mean(most_recent) for loss in self.losses]
        return sum(means) / len(means)

    def __repr__(self):
        rep = f"{self.__class__.__name__}("
        if self.losses:
            losses = ", ".join(repr(l) for l in self.losses)
            rep += f" {self.mean(100):.5f}/{self.mean():.5f} : {losses} "
        return rep + ")"

    class _BaseLoss(nn.Module):
        def __init__(self, weight=1):
            super().__init__()
            self.weight = weight
            self.history = []

        def forward(self, inputs, outputs):
            loss = self.loss(inputs, outputs)
            return loss * self.weight

        def mean(self, most_recent=None):
            history = self.history
            if most_recent:
                history = history[-most_recent:]
            if not history:
                return 0
            return sum(history) / len(history)

        def __repr__(self):
            rep = f"{self.__class__.__name__}["
            if self.weight != 1:
                rep += f" ({self.weight}x)"
            if self.history:
                rep += f" {self.mean(100):.4f}/{self.mean():.4f}"
            return rep + " ]"

    class Supervised(_BaseLoss):
        def loss(self, inputs, outputs):
            gt = inputs["frames"]
            rend = outputs["renders"]

            # (batch, index, channel, y, x)
            gt_rgb = gt[:, :, :3]
            rend_rgb = rend[:, :, :3]
            gt_alpha = gt[:, :, -1:]
            rend_alpha = rend[:, :, -1:]

            # apply weighting only over image dims
            mask = gt_alpha > 0
            dims = (2, 3, 4)

            alpha_loss_in = self._l1(gt_alpha, rend_alpha, mask, dims)
            alpha_loss_out = self._l1(gt_alpha, rend_alpha, ~mask, dims)
            rgb_loss = self._l1(gt_rgb * gt_alpha, rend_rgb * rend_alpha, mask, dims)

            return (alpha_loss_in + alpha_loss_out + rgb_loss) / 3

        def _l1(_, a, b, weights, weight_dims):
            error = (a - b).abs()
            error = (error * weights).sum(weight_dims)
            weights = weights.sum(weight_dims)
            weights = weights + (weights == 0)  # no division by 0
            error /= weights
            return error.mean(1)

    class TemporalConsistency(_BaseLoss):
        def __init__(self, padding, **kwargs):
            super().__init__(**kwargs)
            self.padding = padding

        def loss(self, _, outputs):
            rend = outputs["renders"]
            padding = int(self.padding * rend.shape[-1])

            # (batch, index, offset_y, offset_x)
            zncc = self._zncc(rend[:, :-1], rend[:, 1:], padding)

            # range [-1, 1] -> [1, 0]
            return (1 - zncc.amax((2, 3)).mean(1)) / 2

        def _zncc(self, a, b, padding):
            # (batch, index, channel, y, x)
            image_dims = (2, 3, 4)
            image_shape = a.shape[2:]

            a = self._normalize(a, image_dims)
            b = self._normalize(b, image_dims)

            # reshape to single batch with images as channels
            inputs = a.reshape(1, -1, *image_shape)
            weight = b.reshape(-1, 1, *image_shape)
            padding = (0, padding, padding)

            cc = nn.functional.conv3d(
                inputs,
                weight,
                padding=padding,
                groups=len(weight),
            )
            cc /= image_shape.numel()
            return cc.reshape(*a.shape[:2], *cc.shape[-2:])

        def _normalize(_, tensor, dims):
            mean = tensor.mean(dims, keepdims=True)
            std = tensor.std(dims, unbiased=False, keepdims=True)
            std = std + (std == 0)  # no division by 0
            return (tensor - mean) / std
