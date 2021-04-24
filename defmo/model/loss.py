import torch
import torch.nn as nn


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
