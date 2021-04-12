import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [
            Loss.Supervised(),
            Loss.TemporalConsistency(padding=0.1, weight=10),
        ]

    def forward(self, inputs, **outputs):
        return sum(loss(inputs, outputs) for loss in self.losses)

    def reset(self):
        for loss in self.losses:
            loss.reset()

    def __repr__(self):
        return f"{self.__class__.__name__}( {', '.join(repr(l) for l in self.losses)} )"

    class _BaseLoss(torch.nn.Module):
        def __init__(self, weight=1):
            super().__init__()
            self.weight = weight
            self.history = []

        def forward(self, inputs, outputs):
            loss = self.loss(inputs, outputs)
            self.history.append(loss.item())
            return loss * self.weight

        def reset(self):
            self.history.clear()

        def __repr__(self):
            rep = self.__class__.__name__
            if self.history:
                rep += "[ "
                if self.weight != 1:
                    rep += f"{self.weight} * "
                rep += f"{sum(self.history) / len(self.history):.6f}"
                rep += " ]"
            return rep

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

            return alpha_loss_in + alpha_loss_out + rgb_loss

        def _l1(_, a, b, weights, weight_dims):
            error = (a - b).abs()
            weighted = (error * weights).sum(weight_dims)
            weighted /= weights.sum(weight_dims)
            return weighted.mean()

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
            return (1 - zncc.amax((2, 3)).mean()) / 2

        def _zncc(self, a, b, padding):
            # (batch, index, channel, y, x)
            image_dims = (2, 3, 4)
            image_shape = a.shape[2:]

            a = self._normalize(a, image_dims)
            b = self._normalize(b, image_dims)

            # single batch with images as channels since weights are different for each image
            inputs = a.reshape(1, -1, *image_shape)
            weight = b.reshape(-1, 1, *image_shape)
            padding = (0, padding, padding)

            cc = torch.nn.functional.conv3d(inputs, weight, padding=padding, groups=len(weight))
            cc /= image_shape.numel()
            return cc.reshape(*a.shape[:2], *cc.shape[-2:])

        def _normalize(_, tensor, dims):
            mean = tensor.mean(dims, keepdims=True)
            std = tensor.std(dims, unbiased=False, keepdims=True)
            std += std == 0
            return (tensor - mean) / std
