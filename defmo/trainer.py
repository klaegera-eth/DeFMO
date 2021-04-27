import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR as Scheduler


class Trainer:
    def __init__(
        self,
        model,
        lr=0.001,
        lr_steps=20,
        lr_decay=0.8,
        checkpoint=None,
    ):
        # required for correct operation of torch multiprocessing
        torch.multiprocessing.set_start_method("spawn", force=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model_dp = nn.parallel.DistributedDataParallel(model)
        self.optimizer = Optimizer(model.parameters(), lr=lr)
        self.scheduler = Scheduler(self.optimizer, lr_steps, lr_decay)

        if checkpoint is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.epoch = checkpoint["epochs"]
            self.loss = checkpoint["loss"]
        else:
            self.epoch = 0
            self.loss = {"train": [], "valid": []}

    def train(self, datasets, epochs, batch_size, benchmark=False):
        torch.backends.cudnn.benchmark = benchmark

        datasets = {
            k: DataLoader(
                ds, batch_size, shuffle=True, num_workers=torch.get_num_threads()
            )
            for k, ds in datasets.items()
        }

        print(
            f"Begin training ({self.device}) -",
            f"{torch.get_num_threads()} CPUs,",
            f"{torch.cuda.device_count()} GPUs",
        )
        epochs += self.epoch
        for _ in range(self.epoch, epochs):
            self.epoch += 1
            self.model_dp.train()

            ds = datasets["train"]
            for batch, inputs in enumerate(ds):
                losses = self.model_dp(inputs)
                self.model.loss.record(losses)

                losses.mean().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                print(
                    f"Epoch {self.epoch:0{len(str(epochs))}}/{epochs}",
                    f"Batch {batch + 1:0{len(str(len(ds)))}}/{len(ds)}",
                    self.model.loss,
                )

            self.loss["train"].append(self.model.loss.mean(1000))
            self.scheduler.step()

            with torch.no_grad():
                self.model_dp.eval()

                ds = datasets["valid"]
                for batch, inputs in enumerate(ds):
                    losses = self.model_dp(inputs)
                    self.model.loss.record(losses)

                print(
                    f"Epoch {self.epoch:0{len(str(epochs))}}/{epochs}",
                    f"Validation ({len(ds)} batches)",
                    self.model.loss,
                )

                self.loss["valid"].append(self.model.loss.mean())

                if self.loss["valid"][-1] == min(self.loss["valid"]):
                    self.save("checkpoint_best.pt")

        self.save("checkpoint_end.pt")

    def save(self, filename):
        if not torch.distributed.get_rank():
            print("Saving", filename)
            torch.save(
                {
                    "model": self.model.get_state(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "loss": self.loss,
                    "epochs": self.epoch,
                },
                filename,
            )
