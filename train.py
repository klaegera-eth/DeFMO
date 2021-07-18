import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import defmo.training as tr
from datasets import get_dataset


def main(args):
    pl.seed_everything(args.seed, workers=True)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            tr.callbacks.LogGTvsRenders(),
            tr.callbacks.LogPrediction(),
            tr.callbacks.ContinuousModelCheckpoint(args.checkpoint, "valid_loss"),
        ],
        logger=TensorBoardLogger(
            save_dir="logs",
            name=args.name,
            version=args.version,
            default_hp_metric=False,
        ),
    )

    if args.description:
        trainer.logger.experiment.add_text("description", args.description)

    trainer.fit(
        model=tr.DeFMO.from_args(args),
        datamodule=get_dataset(args.dataset, num_workers=args.dataset_workers),
    )


if __name__ == "__main__":
    parser = tr.DeFMOArgParser()

    args, _ = parser.parse_known_args()

    if parser.load_checkpoint_defaults(args):
        print(f"Checkpoint Loaded: {args.checkpoint}")
        tr.DeFMO.add_model_specific_args(parser, required=False)
    else:
        print("Specified checkpoint does not exist, starting new session.")
        tr.DeFMO.add_model_specific_args(parser, required=True)

    main(parser.parse_args())
