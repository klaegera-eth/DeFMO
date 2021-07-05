import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl


class DeFMOArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        pl.Trainer.add_argparse_args(self)
        self.set_defaults(
            gpus=-1,
            accelerator="ddp",
            plugins="ddp_find_unused_parameters_false",
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            checkpoint_callback=True,
        )

        self.add_argument("--dataset", required=True)
        self.add_argument("--dataset_workers", type=int, default=1)
        self.add_argument("--epochs", type=int, required=True)
        self.add_argument("--checkpoint")
        self.add_argument("--name", default="noname")
        self.add_argument("--version", default="noversion")
        self.add_argument("--description")

    def load_checkpoint_defaults(self, args):
        if os.path.isfile(args.checkpoint):
            cp = torch.load(args.checkpoint, map_location="cpu")
            self.set_defaults(
                max_epochs=cp["epoch"] + args.epochs,
                resume_from_checkpoint=args.checkpoint,
                **cp["hyper_parameters"],
            )
            return True
        return False

    def parse_known_args(self, *args, **kwargs):
        args, rest = super().parse_known_args(*args, **kwargs)
        return self._process_args(args), rest

    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        return self._process_args(args)

    def _process_args(self, args):
        if not args.checkpoint:
            args.checkpoint = f"checkpoints/{args.name}/{args.version}.ckpt"

        if not self.get_default("max_epochs"):
            self.set_defaults(max_epochs=args.epochs)
        args.max_epochs = args.max_epochs or args.epochs

        return args
