import sys, os
import torch

from defmo.training import Trainer, ZipDataset
from defmo.training.modules import Model, Loss
from defmo.utils import ZipLoader


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

    models = dict(
        encoder="resnet_gn_nomaxpool",
        renderer="resnet_gn",
    )
    losses = [
        Loss.Supervised(),
        # Loss.TemporalConsistency(padding=0.1),
    ]

    _, name, epochs = sys.argv

    chkpt = None
    if os.path.isfile(name + ".pt"):
        chkpt = torch.load(name + ".pt", map_location="cpu")
        models = chkpt["models"]

    model = Model(**models, losses=losses)
    if chkpt:
        model.load_state_dict(chkpt["model_state"])

    trainer = Trainer(name, model)
    if chkpt:
        trainer.load(chkpt)

    trainer.train(datasets, epochs=int(epochs), batch_size=4, benchmark=False)


if __name__ == "__main__":

    if "RANK" not in os.environ:
        # launcher process

        if len(sys.argv) != 3:
            sys.exit("usage: train.py <name> <epochs>")

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
