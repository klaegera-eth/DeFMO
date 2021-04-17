from defmo import dataset, utils, train
from defmo.loss import Loss


if __name__ == "__main__":

    ds_train = dataset.ZipDataset(
        "data/fmo_3_24_v1.zip",
        utils.ZipLoader("data/vot2018.zip", balance_subdirs=True),
        item_range=(0, 0.9),
    )

    ds_val = dataset.ZipDataset(
        "data/fmo_3_24_v1.zip",
        utils.ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
        item_range=(0.9, 1),
    )

    losses = [
        Loss.Supervised(),
        # Loss.TemporalConsistency(padding=0.1),
    ]

    train.train(ds_train, ds_val, 1, losses, batch_size=3)
