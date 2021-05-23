import torch

from .utils import to_tensor_stack


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, fmo_loader, background_adder):
        self._fmo_loader = fmo_loader
        self._bg_adder = background_adder

    def __len__(self):
        return len(self._fmo_loader)

    def __getitem__(self, index):
        blurs, frames = self._fmo_loader[index]
        imgs, bgs = self._bg_adder(blurs)

        return {
            "imgs": to_tensor_stack(imgs),
            "blurs": to_tensor_stack(blurs),
            "frames": to_tensor_stack(frames),
            "bgs": to_tensor_stack(bgs),
        }
