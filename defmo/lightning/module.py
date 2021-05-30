import torch
import pytorch_lightning as pl

from pytorch_lightning.utilities import rank_zero_only

import defmo.training.modules as mod


class DeFMO(pl.LightningModule):
    def __init__(self, encoder, renderer, losses):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = mod.Encoder(encoder)
        self.renderer = mod.Renderer(renderer)
        self.loss = mod.Loss(losses)

    @classmethod
    def add_model_specific_args(cls, parser, required=True):
        group = parser.add_argument_group(cls.__name__)
        group.add_argument("--encoder", required=required)
        group.add_argument("--renderer", required=required)
        group.add_argument("--losses", nargs="+", required=required)

    @classmethod
    def from_args(cls, args, **kwargs):
        return cls(args.encoder, args.renderer, args.losses, **kwargs)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"valid_loss": float("inf")})

    def forward(self, imgs, n_frames):
        latent = self.encoder(imgs)
        renders = self.renderer(latent, n_frames)
        return dict(latent=latent, renders=renders)

    def step(self, inputs):
        n_frames = inputs["frames"].shape[1]
        outputs = self(inputs["imgs"], n_frames)
        return self.loss(inputs, outputs).mean(0), outputs

    def training_step(self, inputs, _):
        loss, _ = self.step(inputs)
        self.loss.log("train_loss", loss, self.log, sync_dist=True)
        loss = loss.mean()
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, inputs, batch_idx):
        loss, outputs = self.step(inputs)
        self.loss.log("valid_loss", loss, self.log, sync_dist=True)
        loss = loss.mean()
        self.log("valid_loss", loss, sync_dist=True)

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
