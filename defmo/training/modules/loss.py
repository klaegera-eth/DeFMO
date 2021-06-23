import torch
import torch.nn as nn
import torchmetrics.functional as metrics


class Loss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = [
            loss if isinstance(loss, Loss._BaseLoss) else getattr(Loss, loss)()
            for loss in losses
        ]

    def forward(self, inputs, outputs):
        losses = [loss(inputs, outputs) for loss in self.losses]
        return torch.stack(losses, 1) if losses else torch.Tensor()

    def log(self, name, loss, log_fn, **log_fn_kwargs):
        with torch.no_grad():
            for val, cls in zip(loss, self.losses):
                log_fn(f"{name}/{cls._get_name()}", val / cls.weight, **log_fn_kwargs)

    def __repr__(self):
        return f"{self._get_name()}({', '.join(repr(loss) for loss in self.losses)})"

    class _BaseLoss(nn.Module):
        def __init__(self, weight=1):
            super().__init__()
            self.weight = weight

        def forward(self, inputs, outputs):
            loss = self.loss(inputs, outputs)
            return loss * self.weight

        def __repr__(self):
            return f"{self._get_name()}(weight={self.weight})"

        def _weighted_mean(_, x, weights, dims):
            x = (x * weights).sum(dims)
            weights = weights.sum(dims)
            weights = weights + (weights == 0)  # no division by 0
            return x / weights

    class _SupervisedBase(_BaseLoss):
        def loss(self, inputs, outputs):
            gt = inputs["frames"]
            rend = outputs["renders"]
            return self.supervised(gt, rend)

        def _split(_, *tensors):
            # split RGB / alpha
            # (batch, index, channel, y, x)
            return [(t[:, :, :3], t[:, :, -1:]) for t in tensors]

    class Supervised(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            alpha_l1 = (gt_alpha - rend_alpha).abs()
            rgb_l1 = (gt_rgb * gt_alpha - rend_rgb * rend_alpha).abs()

            # apply weighting only over image dims
            mask = gt_alpha > 0
            dims = (2, 3, 4)

            alpha_loss_in = self._weighted_mean(alpha_l1, mask, dims)
            alpha_loss_out = self._weighted_mean(alpha_l1, ~mask, dims)
            rgb_loss = self._weighted_mean(rgb_l1, mask, dims)

            return (alpha_loss_in + alpha_loss_out + rgb_loss).mean(1) / 3

    class SupervisedL2(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            alpha_l2 = (gt_alpha - rend_alpha) ** 2
            rgb_l2 = (gt_rgb - rend_rgb) ** 2

            alpha_loss = alpha_l2.mean((2, 3, 4))
            rgb_loss = self._weighted_mean(rgb_l2, gt_alpha, (2, 3, 4))
            return (alpha_loss + rgb_loss).mean(1) / 2

    class AlphaDice(_SupervisedBase):
        def supervised(self, gt, rend):
            (_, gt_a), (_, r_a) = self._split(gt, rend)

            return 1 - 2 * (
                (r_a * gt_a).sum((1, 2, 3, 4))
                / ((r_a * r_a).sum((1, 2, 3, 4)) + (gt_a * gt_a).sum((1, 2, 3, 4)))
            )

    class RGBSSIM(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            gt, rend = gt_rgb * gt_alpha, rend_rgb * rend_alpha
            return 1 - torch.stack([metrics.ssim(r, g) for r, g in zip(rend, gt)])

    class SupervisedSSIM(_SupervisedBase):
        def supervised(self, gt, rend):
            return 1 - torch.stack([metrics.ssim(r, g) for r, g in zip(rend, gt)])

    class SupervisedL1AlphaSSIM(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            alpha_l1 = (gt_alpha - rend_alpha).abs()
            alpha_loss = alpha_l1.mean((1, 2, 3, 4))

            rgb_loss = 1 - torch.stack(
                [
                    metrics.ssim(r, g)
                    for r, g in zip(rend_rgb * rend_alpha, gt_rgb * gt_alpha)
                ]
            )

            return (alpha_loss + rgb_loss) / 2

    class TemporalConsistency(_BaseLoss):
        def __init__(self, padding=0.1, **kwargs):
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
