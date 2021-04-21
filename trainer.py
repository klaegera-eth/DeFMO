from defmo.train import train
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

    model = Model(
        n_frames=datasets["training"].params["n_frames"],
        encoder="v2",
        renderer="resnet",
        losses=[
            Loss.Supervised(),
            # Loss.TemporalConsistency(padding=0.1),
        ],
    )

    train(datasets, model, epochs=1, batch_size=3)
