import sys, os
import torch

from defmo import Trainer, ZipDataset, ZipLoader
from defmo.model import Model, Loss


if __name__ == "__main__":

    datasets = {
        "train": ZipDataset(
            "data/fmo_3_24_v1.zip",
            ZipLoader("data/vot2018.zip", balance_subdirs=True),
            item_range=(0, 0.9),
        ),
        "valid": ZipDataset(
            "data/fmo_3_24_v1.zip",
            ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
            item_range=(0.9, 1),
        ),
    }

    losses = [
        Loss.Supervised(),
        # Loss.TemporalConsistency(padding=0.1),
    ]

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        chkp = torch.load(sys.argv[1], map_location="cpu")
        model = Model(losses, checkpoint=chkp["model"])
        trainer = Trainer(model, checkpoint=chkp)
        print(f"Loaded: {chkp['epochs']} epochs, {chkp['loss']['valid'][-1]:.5f} loss")
    else:
        model = Model(losses, encoder="v2", renderer="resnet")
        trainer = Trainer(model)
        print("Loaded default model")

    trainer.train(datasets, epochs=1, batch_size=3)
