import torch

from defmo import dataset, utils, train, models
from defmo.loss import Loss


if __name__ == "__main__":

    datasets = {
        "training": dataset.ZipDataset(
            "data/fmo_3_24_v1.zip",
            utils.ZipLoader("data/vot2018.zip", balance_subdirs=True),
            item_range=(0, 0.9),
        ),
        "validation": dataset.ZipDataset(
            "data/fmo_3_24_v1.zip",
            utils.ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
            item_range=(0.9, 1),
        ),
    }

    modules = {
        "encoder": models.Encoder(),
        "renderer": models.Renderer(datasets["training"].params["n_frames"]),
    }

    def step(modules, inputs):
        imgs = inputs["imgs"][:, 1], inputs["imgs"][:, 2]
        latent = modules["encoder"](torch.cat(imgs, 1))
        renders = modules["renderer"](latent)
        return {"renders": renders}

    loss = Loss(
        [
            Loss.Supervised(),
            # Loss.TemporalConsistency(padding=0.1),
        ]
    )

    train.train(datasets, modules, step, loss, epochs=1, batch_size=3)
