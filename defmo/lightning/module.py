import pytorch_lightning as pl

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

    def forward(self, imgs, n_frames):
        latent = self.encoder(imgs)
        renders = self.renderer(latent, n_frames)
        return dict(latent=latent, renders=renders)

    def step(self, inputs):
        n_frames = inputs["frames"].shape[1]
        outputs = self(inputs["imgs"], n_frames)
        return self.loss(inputs, outputs).mean(0)

    def training_step(self, inputs, _):
        loss = self.step(inputs)
        self.loss.log("train_loss", loss, self.log)
        loss = loss.mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, inputs, _):
        loss = self.step(inputs)
        self.loss.log("valid_loss", loss, self.log)
        loss = loss.mean()
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        from torch.optim import Adam

        return Adam(self.parameters())

    def optimizer_zero_grad(self, _, __, optimizer, ___):
        optimizer.zero_grad(set_to_none=True)
