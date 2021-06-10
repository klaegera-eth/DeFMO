import os
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from defmo.lightning import DeFMO, ContinuousModelCheckpoint

from datasets import get_dataset


def main(args):
    callbacks = []
    if args.checkpoint:
        if args.checkpoint == "auto":
            cpdir = f"checkpoints/{args.name}/{args.version}"
        else:
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
    data = get_dataset(args.dataset, args.dataset_workers)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.name,
        version=args.version,
        default_hp_metric=False,
    )

    if args.description:
        logger.experiment.add_text("description", args.description)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger)

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(ArgumentParser())
    parser.set_defaults(
        gpus=-1,
        accelerator="ddp",
        plugins="ddp_find_unused_parameters_false",
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--name", default="noname")
    parser.add_argument("--version", default="noversion")
    parser.add_argument("--description")

    trainer_args, _ = parser.parse_known_args()
    chkpt = trainer_args.checkpoint
    parser.set_defaults(
        max_epochs=trainer_args.epochs,
        checkpoint_callback=not not chkpt,
    )

    model_args_required = True
    if chkpt:
        if chkpt == "auto":
            chkpt = f"checkpoints/{trainer_args.name}/{trainer_args.version}/last.ckpt"
        if os.path.isfile(chkpt):
            print("Resuming checkpoint", chkpt)
            cp = torch.load(chkpt, map_location="cpu")
            parser.set_defaults(
                max_epochs=cp["epoch"] + trainer_args.epochs,
                resume_from_checkpoint=chkpt,
                **cp["hyper_parameters"],
            )
            model_args_required = False
        else:
            print("Specified checkpoint does not exist, starting new session.")

    DeFMO.add_model_specific_args(parser, required=model_args_required)

    main(parser.parse_args())
