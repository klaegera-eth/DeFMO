import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR as Scheduler

from defmo.models import Encoder, Renderer
from defmo.loss import Loss


def train(
    data_train,
    data_val,
    epochs,
    losses,
    batch_size=20,
    lr=0.001,
    lr_steps=1000,
    lr_decay=0.5,
):

    # required for correct operation of torch multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.benchmark = True

    gen_train, gen_val = (
        DataLoader(data, batch_size, shuffle=True, num_workers=torch.get_num_threads())
        for data in (data_train, data_val)
    )

    encoder, renderer, loss = (
        nn.DataParallel(module).to(device)
        for module in (
            Encoder(),
            Renderer(data_train.params["n_frames"]),
            Loss(losses),
        )
    )

    optimizer = Optimizer(
        (p for m in (encoder, renderer) for p in m.parameters()),
        lr=lr,
    )
    scheduler = Scheduler(optimizer, lr_steps, lr_decay)

    best_loss_val = float("inf")

    print("Begin")
    for epoch in range(epochs):
        encoder.train(), renderer.train(), loss.module.reset()

        for batch, inputs in enumerate(gen_train):

            imgs = inputs["imgs"][:, 1], inputs["imgs"][:, 2]
            latent = encoder(torch.cat(imgs, 1))
            renders = renderer(latent)
            loss(inputs, renders=renders).backward()

            optimizer.step()
            optimizer.zero_grad()

            print(
                f"Epoch {epoch + 1:0{len(str(epochs))}}/{epochs}",
                f"Batch {batch + 1:0{len(str(len(gen_train)))}}/{len(gen_train)}",
                loss.module,
            )

        loss_train = loss.module.mean()
        scheduler.step()

        with torch.no_grad():
            encoder.eval(), renderer.eval(), loss.module.reset()

            for batch, inputs in enumerate(gen_val):

                imgs = inputs["imgs"][:, 1], inputs["imgs"][:, 2]
                latent = encoder(torch.cat(imgs, 1))
                renders = renderer(latent)
                loss(inputs, renders=renders)

            print(
                f"Epoch {epoch + 1:0{len(str(epochs))}}/{epochs}",
                f"Validation ({len(gen_val)} batches)",
                loss.module,
            )

            loss_val = loss.module.mean()

        checkpoint = {
            "modules": {
                "encoder": encoder.module,
                "renderer": renderer.module,
                "loss": loss.module,
            },
            "scores": (loss_train, loss_val),
            "epochs": epoch + 1,
        }

        if loss_val < best_loss_val:
            best_loss_val = loss_val
            print(f"Saving (train: {loss_train:.6f}, valid: {loss_val:.6f})")
            torch.save(checkpoint, "checkpoint_best.pt")

    print(f"Saving (train: {loss_train:.6f}, valid: {loss_val:.6f})")
    torch.save(checkpoint, "checkpoint_end.pt")
