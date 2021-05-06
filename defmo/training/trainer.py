import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.nn.modules.batchnorm import SyncBatchNorm

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau as Scheduler


class Trainer:
    def __init__(self, name, model):
        # required for correct operation of torch multiprocessing
        torch.multiprocessing.set_start_method("spawn", force=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.io = torch.distributed.get_rank() == 0

        self.name = name
        self.model = model

        self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = self.model.to(self.device)
        self.model_dp = nn.parallel.DistributedDataParallel(self.model)

        self.optimizer = Optimizer(self.model.parameters())
        self.scheduler = Scheduler(self.optimizer, verbose=self.io)

        self.epoch = 0
        self.loss = {"train": [], "valid": []}

    def load(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scheduler.verbose = self.io
        self.epoch = checkpoint["epochs"]
        self.loss = checkpoint["loss"]

    def train(self, datasets, epochs, batch_size, benchmark=False):
        torch.backends.cudnn.benchmark = benchmark

        datasets = {
            k: DataLoader(
                ds,
                batch_size,
                sampler=DistributedSampler(ds),
                num_workers=torch.get_num_threads(),
            )
            for k, ds in datasets.items()
        }

        if self.io:
            print(
                f"Begin training ({self.device}) -",
                f"{torch.distributed.get_world_size()} processes x (",
                f"{torch.get_num_threads()} CPUs,",
                f"{torch.cuda.device_count()} GPUs )",
            )

        for _ in range(epochs):
            self.epoch += 1
            for ds in datasets.values():
                ds.sampler.set_epoch(self.epoch)

            self.model_dp.train()
            self.process(datasets["train"], backward=True, verbose=self.io)
            self.loss["train"].append(self.model.loss.mean(most_recent=1000))

            with torch.no_grad():

                self.model_dp.eval()
                self.process(datasets["valid"], backward=False)
                self.loss["valid"].append(self.model.loss.mean())

                self.scheduler.step(self.loss["valid"][-1])

                if self.io:
                    print(
                        f"Epoch {self.epoch:02}",
                        f"Validation ({len(ds)} batches)",
                        self.model.loss,
                    )

                    self.save()

                    if self.loss["valid"][-1] == min(self.loss["valid"]):
                        self.save(suffix="_best")

    def process(self, dataset, backward, verbose=False):
        for batch, inputs in enumerate(dataset):
            losses = self.model_dp(inputs)
            self.model.loss.record(losses)

            if backward:
                self.optimizer.zero_grad()
                losses.mean().backward()
                self.optimizer.step()

            if verbose:
                print(
                    f"Epoch {self.epoch:02}",
                    f"Batch {batch + 1:0{len(str(len(dataset)))}}/{len(dataset)}",
                    self.model.loss,
                )

    def save(self, suffix=""):
        filename = f"{self.name}{suffix}.pt"
        print("Saving", filename)
        torch.save(
            {
                "models": self.model.models,
                "model_state": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "loss": self.loss,
                "epochs": self.epoch,
            },
            filename,
        )
