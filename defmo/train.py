import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR as Scheduler


def train(
    datasets,
    model,
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

    model_dp = nn.DataParallel(model).to(device)
    optimizer = Optimizer(model.parameters(), lr=lr)
    scheduler = Scheduler(optimizer, lr_steps, lr_decay)

    best_loss_val = float("inf")

    print(
        f"Begin training ({device}) -",
        f"{torch.get_num_threads()} CPUs,",
        f"{torch.cuda.device_count()} GPUs",
    )
    for epoch in range(epochs):
        model_dp.train()

        ds = datasets["training"]
        for batch, inputs in enumerate(ds):
            losses = model_dp(inputs)
            model.loss.backward(losses)

            optimizer.step()
            optimizer.zero_grad()

            print(
                f"Epoch {epoch + 1:0{len(str(epochs))}}/{epochs}",
                f"Batch {batch + 1:0{len(str(len(ds)))}}/{len(ds)}",
                model.loss,
            )

        loss_train = model.loss.mean()
        scheduler.step()

        with torch.no_grad():
            model_dp.eval()

            ds = datasets["validation"]
            for batch, inputs in enumerate(ds):
                losses = model_dp(inputs)
                model.loss.backward(losses)

            print(
                f"Epoch {epoch + 1:0{len(str(epochs))}}/{epochs}",
                f"Validation ({len(ds)} batches)",
                model.loss,
            )

            loss_val = model.loss.mean()

        checkpoint = {
            "model": model,
            "scores": (loss_train, loss_val),
            "epochs": epoch + 1,
        }

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            print(f"Saving (train: {loss_train:.6f}, valid: {loss_val:.6f})")
            torch.save(checkpoint, "checkpoint_best.pt")

    print(f"Saving (train: {loss_train:.6f}, valid: {loss_val:.6f})")
    torch.save(checkpoint, "checkpoint_end.pt")
