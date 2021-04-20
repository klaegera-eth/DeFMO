import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR as Scheduler


def train(
    datasets,
    modules,
    step,
    loss,
    epochs,
    batch_size=20,
    lr=0.001,
    lr_steps=4,
    lr_decay=0.8,
    benchmark=False,
):

    # required for correct operation of torch multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = benchmark

    datasets = {
        k: DataLoader(ds, batch_size, shuffle=True, num_workers=torch.get_num_threads())
        for k, ds in datasets.items()
    }

    modules_ = {k: nn.DataParallel(m).to(device) for k, m in modules.items()}
    loss_ = nn.DataParallel(loss).to(device)

    optimizer = Optimizer((p for m in modules_.values() for p in m.parameters()), lr=lr)
    scheduler = Scheduler(optimizer, lr_steps, lr_decay)

    best_loss_val = float("inf")

    print(
        f"Begin training ({device}) -",
        f"{torch.get_num_threads()} CPUs,",
        f"{torch.cuda.device_count()} GPUs",
    )
    for epoch in range(epochs):
        loss.reset()
        for m in modules_.values():
            m.train()

        ds = datasets["training"]
        for batch, inputs in enumerate(ds):
            outputs = step(modules_, inputs)
            loss_(inputs, outputs).mean().backward()

            optimizer.step()
            optimizer.zero_grad()

            print(
                f"Epoch {epoch + 1:0{len(str(epochs))}}/{epochs}",
                f"Batch {batch + 1:0{len(str(len(ds)))}}/{len(ds)}",
                loss,
            )

        loss_train = loss.mean()
        scheduler.step()

        with torch.no_grad():
            loss.reset()
            for m in modules_.values():
                m.eval()

            ds = datasets["validation"]
            for batch, inputs in enumerate(ds):
                outputs = step(modules_, inputs)
                loss_(inputs, outputs)

            print(
                f"Epoch {epoch + 1:0{len(str(epochs))}}/{epochs}",
                f"Validation ({len(ds)} batches)",
                loss,
            )

            loss_val = loss.mean()

        checkpoint = {
            "modules": modules,
            "loss": loss,
            "scores": (loss_train, loss_val),
            "epochs": epoch + 1,
        }

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            print(f"Saving (train: {loss_train:.6f}, valid: {loss_val:.6f})")
            torch.save(checkpoint, "checkpoint_best.pt")

    print(f"Saving (train: {loss_train:.6f}, valid: {loss_val:.6f})")
    torch.save(checkpoint, "checkpoint_end.pt")
