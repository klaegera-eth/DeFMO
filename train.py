import sys, os
import torch

from defmo import Trainer, ZipDataset, ZipLoader
from defmo.model import Model, Loss


def train():
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
    else:
        model = Model(losses, encoder="v2", renderer="resnet")
        trainer = Trainer(model)

    trainer.train(datasets, epochs=1, batch_size=3)


if __name__ == "__main__":

    if "RANK" not in os.environ:
        # launcher process

        import torch.distributed.launch

        sys.argv = [
            "<launcher>",
            "--use_env",
            "--nproc_per_node",
            str(torch.cuda.device_count()),
        ] + sys.argv

        torch.distributed.launch.main()

    else:
        # worker process

        backend = "nccl" if torch.distributed.is_nccl_available() else "gloo"
        torch.distributed.init_process_group(backend)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.distributed.get_rank())

        train()
