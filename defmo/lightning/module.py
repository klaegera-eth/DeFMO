import torch
import pytorch_lightning as pl

from pytorch_lightning.utilities import rank_zero_only

import defmo.training.modules as mod


class DeFMO(pl.LightningModule):
    def __init__(self, encoder, renderer, losses, comparison_losses=[]):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = mod.Encoder(encoder)
        self.renderer = mod.Renderer(renderer)
        self.loss = mod.Loss(losses)
        self.comparison_loss = mod.Loss(comparison_losses)

    @classmethod
    def add_model_specific_args(cls, parser, required=True):
        group = parser.add_argument_group(cls.__name__)
        group.add_argument("--encoder", required=required)
        group.add_argument("--renderer", required=required)
        group.add_argument("--losses", nargs="+", required=required)
        group.add_argument("--comparison-losses", nargs="+", default=[])

    @classmethod
    def from_args(cls, args, **kwargs):
        return cls(
            args.encoder,
            args.renderer,
            args.losses,
            args.comparison_losses,
            **kwargs,
        )

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"valid_loss": float("inf")})

    def forward(self, inputs, n_frames=None):
        if isinstance(inputs, dict):
            if not n_frames:
                n_frames = inputs["frames"].shape[1]
            inputs = inputs["imgs"]
        latent = self.encoder(inputs)
        renders = self.renderer(latent, n_frames)
        return dict(latent=latent, renders=renders)

    def step(self, inputs, log_name):
        outputs = self(inputs)
        loss = self.loss(inputs, outputs).mean(0)

        self.loss.log(log_name, loss, self.log, sync_dist=True)
        loss = loss.mean()
        self.log(log_name, loss, sync_dist=True)

        with torch.no_grad():
            closs = self.comparison_loss(inputs, outputs).mean(0)
            self.comparison_loss.log(
                log_name, closs, self.log, sync_dist=True, prog_bar=True
            )

        return loss, outputs

    def training_step(self, inputs, _):
        loss, _ = self.step(inputs, "train_loss")

        return loss

    def validation_step(self, inputs, batch_idx):
        loss, outputs = self.step(inputs, "valid_loss")

        if batch_idx == 0:
            self._log_gt_vs_renders(gt=inputs["frames"], renders=outputs["renders"])

        return loss

    @rank_zero_only
    def _log_gt_vs_renders(self, gt, renders):
        vids = torch.cat((gt, renders), -1)[:5]
        vids = torch.cat(tuple(iter(vids)), -2)
        rgb, alpha = vids[:, :3], vids[:, 3:]
        vids = alpha * rgb + (1 - alpha)
        self.logger.experiment.add_video(
            "gt_vs_renders", vids[None], fps=24, global_step=self.current_epoch
        )

    def configure_optimizers(self):
        from torch.optim import Adam

        return Adam(self.parameters())

    def optimizer_zero_grad(self, _, __, optimizer, ___):
        optimizer.zero_grad(set_to_none=True)
