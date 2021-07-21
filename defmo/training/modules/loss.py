import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as metrics

import pytorch_msssim as ssim


class Loss(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = [
            loss if isinstance(loss, Loss._BaseLoss) else self.parse(loss)
            for loss in losses
        ]

    def parse(self, loss_str):
        # example: "MyLoss:param1=1,param2=myval,param3=4.5"
        if ":" in loss_str:
            loss, params = loss_str.split(":")
            loss = getattr(Loss, loss)
            params = dict(p.split("=") for p in params.split(","))
            for k, v in params.items():
                try:
                    params[k] = float(v)
                    params[k] = int(v)
                except:
                    pass
        else:
            loss = getattr(Loss, loss_str)
            params = {}
        return loss(**params)

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

    class RGBL2(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            rgb_l2 = (gt_rgb * gt_alpha - rend_rgb * rend_alpha) ** 2
            return self._weighted_mean(rgb_l2, gt_alpha, (2, 3, 4)).mean(1)

    class RGBL2GT(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, _) = self._split(gt, rend)

            rgb_l2 = (gt_rgb * gt_alpha - rend_rgb * gt_alpha) ** 2
            return self._weighted_mean(rgb_l2, gt_alpha, (2, 3, 4)).mean(1)

    class RGBL2Direct(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, _) = self._split(gt, rend)

            rgb_l2 = (gt_rgb - rend_rgb) ** 2
            return self._weighted_mean(rgb_l2, gt_alpha, (2, 3, 4)).mean(1)

    class RGBL1(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            rgb_l1 = (gt_rgb * gt_alpha - rend_rgb * rend_alpha).abs()
            return self._weighted_mean(rgb_l1, gt_alpha, (2, 3, 4)).mean(1)

    class RGBL1GT(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, _) = self._split(gt, rend)

            rgb_l1 = (gt_rgb * gt_alpha - rend_rgb * gt_alpha).abs()
            return self._weighted_mean(rgb_l1, gt_alpha, (2, 3, 4)).mean(1)

    class RGBL1Direct(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, _) = self._split(gt, rend)

            rgb_l1 = (gt_rgb - rend_rgb).abs()
            return self._weighted_mean(rgb_l1, gt_alpha, (2, 3, 4)).mean(1)

    class AlphaL2(_SupervisedBase):
        def supervised(self, gt, rend):
            (_, gt_a), (_, r_a) = self._split(gt, rend)

            alpha_l2 = (gt_a - r_a) ** 2
            return alpha_l2.mean((1, 2, 3, 4))

    class AlphaL1(_SupervisedBase):
        def supervised(self, gt, rend):
            (_, gt_a), (_, r_a) = self._split(gt, rend)

            alpha_l1 = (gt_a - r_a).abs()
            return alpha_l1.mean((1, 2, 3, 4))

    class AlphaEntropy(_SupervisedBase):
        def supervised(self, gt, rend):
            (_, gt_a), (_, r_a) = self._split(gt, rend)

            alpha_bce = F.binary_cross_entropy(r_a, gt_a.round(), reduction="none")
            return alpha_bce.mean((1, 2, 3, 4))

    class AlphaDice(_SupervisedBase):
        def supervised(self, gt, rend):
            (_, gt_a), (_, r_a) = self._split(gt, rend)

            return 1 - 2 * (
                (r_a * gt_a).sum((1, 2, 3, 4))
                / ((r_a * r_a).sum((1, 2, 3, 4)) + (gt_a * gt_a).sum((1, 2, 3, 4)))
            )

    class AlphaJaccard(_SupervisedBase):
        def supervised(self, gt, rend):
            (_, gt_a), (_, r_a) = self._split(gt, rend)

            I = (r_a * gt_a).sum((1, 2, 3, 4))
            U = (r_a ** 2 + gt_a ** 2).sum((1, 2, 3, 4))

            return 1 - (I + 1) / (U - I + 1)

    class SSIM(_SupervisedBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ssim = ssim.SSIM(data_range=1)

        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            gt, rend = gt_rgb * gt_alpha, rend_rgb * rend_alpha
            return 1 - torch.stack([self.ssim(r, g) for r, g in zip(rend, gt)])

    class MSSSIM(_SupervisedBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ms_ssim = ssim.MS_SSIM(data_range=1)

        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            gt, rend = gt_rgb * gt_alpha, rend_rgb * rend_alpha
            return 1 - torch.stack([self.ms_ssim(r, g) for r, g in zip(rend, gt)])

    class PSNR(_SupervisedBase):
        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, rend_alpha) = self._split(gt, rend)

            gt, rend = gt_rgb * gt_alpha, rend_rgb * rend_alpha
            return -torch.stack(
                [metrics.psnr(r, g, data_range=1) for r, g in zip(rend, gt)]
            )

    class RGBSSIMGT(_SupervisedBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ssim = ssim.SSIM(data_range=1)

        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, _) = self._split(gt, rend)

            gt, rend = gt_rgb * gt_alpha, rend_rgb * gt_alpha
            return 1 - torch.stack([self.ssim(r, g) for r, g in zip(rend, gt)])

    class RGBMSSSIMGT(_SupervisedBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ms_ssim = ssim.MS_SSIM(data_range=1)

        def supervised(self, gt, rend):
            (gt_rgb, gt_alpha), (rend_rgb, _) = self._split(gt, rend)

            gt, rend = gt_rgb * gt_alpha, rend_rgb * gt_alpha
            return 1 - torch.stack([self.ms_ssim(r, g) for r, g in zip(rend, gt)])

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
