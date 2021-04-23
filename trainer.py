import sys
import torch

from defmo.train import Trainer
from defmo.dataset import ZipDataset
from defmo.utils import ZipLoader
from defmo.models import Model, Loss


if __name__ == "__main__":

    datasets = {
        "training": ZipDataset(
            "data/fmo_3_24_v1.zip",
            ZipLoader("data/vot2018.zip", balance_subdirs=True),
            item_range=(0, 0.9),
        ),
        "validation": ZipDataset(
            "data/fmo_3_24_v1.zip",
            ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
            item_range=(0.9, 1),
        ),
    }

    losses = [
        Loss.Supervised(),
        # Loss.TemporalConsistency(padding=0.1),
    ]

    try:
        chkp = torch.load(sys.argv[1], map_location="cpu")
        model = Model(losses, checkpoint=chkp["model"])
        trainer = Trainer(model, checkpoint=chkp)
        print(f"Loaded {sys.argv[1]}: {chkp['epochs']} epochs, {chkp['loss']:.5f} loss")
    except:
        model = Model(losses, encoder="v2", renderer="resnet")
        trainer = Trainer(model)
        print("Loaded default model")

    trainer.train(datasets, epochs=1, batch_size=3)
