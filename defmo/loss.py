import torch


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [Loss.Supervised()]

    def forward(self, inputs, **outputs):
        return sum(loss(inputs, outputs) for loss in self.losses)

    class _BaseLoss(torch.nn.Module):
        def forward(self, inputs, outputs):
            return self.loss(inputs, outputs)

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
