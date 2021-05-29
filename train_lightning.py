import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from defmo.lightning import DeFMO, ContinuousModelCheckpoint

from datasets import get_dataset


def main(args):
    callbacks = []
    if args.checkpoint:
        cpdir, _ = os.path.split(os.path.realpath(args.checkpoint))
        callbacks.append(
            ContinuousModelCheckpoint(
                dirpath=cpdir,
                filename="best",
                save_last=True,
                monitor="valid_loss",
            )
        )

    model = DeFMO.from_args(args)
    data = get_dataset(args.dataset)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(ArgumentParser())
    parser.set_defaults(
        gpus=-1,
        accelerator="ddp",
        plugins="ddp_find_unused_parameters_false",
    )

    parser.add_argument("--checkpoint")
    trainer_args, _ = parser.parse_known_args()
    chkpt = trainer_args.checkpoint
    parser.set_defaults(checkpoint_callback=not not chkpt)
    model_args_required = True
    if chkpt:
        if os.path.isfile(chkpt):
            hparams = torch.load(chkpt, map_location="cpu")["hyper_parameters"]
            parser.set_defaults(**hparams, resume_from_checkpoint=chkpt)
            model_args_required = False
        else:
            print("Specified checkpoint does not exist, starting new session.")

    DeFMO.add_model_specific_args(parser, required=model_args_required)

    parser.add_argument("--dataset", required=True)

    main(parser.parse_args())
