import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR as Scheduler

from defmo.models import Encoder, Renderer
from defmo.loss import Loss


def train(data_train, data_val, epochs, batch_size=20, lr=0.001, lr_steps=1000, lr_decay=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    gen_train = DataLoader(data_train, batch_size, shuffle=True, drop_last=True, num_workers=torch.get_num_threads())
    gen_val = DataLoader(data_val, batch_size, shuffle=True, drop_last=True, num_workers=torch.get_num_threads())

    encoder = nn.DataParallel(Encoder()).to(device)
    renderer = nn.DataParallel(Renderer(data_train.params["n_frames"])).to(device)
    loss = nn.DataParallel(Loss()).to(device)

    optimizer = Optimizer(list(encoder.parameters()) + list(renderer.parameters()), lr=lr)
    scheduler = Scheduler(optimizer, lr_steps, lr_decay)

    for epoch in range(epochs):
        encoder.train(), renderer.train()

        for batch in gen_train:
            latent = encoder(batch["imgs_short_cat"])
            renders = renderer(latent)

            loss(batch, renders=renders).backward()

            optimizer.step()
            optimizer.zero_grad()

            print(loss.module)

        scheduler.step()
        loss.module.reset()
