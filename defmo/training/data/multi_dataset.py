import random
from torch.utils.data import Dataset

from .utils import alpha_composite, to_tensor_stack


class MultiDataset(Dataset):
    def __init__(self, fmo_loader, background_adder, num_objects=1):
        self._fmo_loader = fmo_loader
        self._bg_adder = background_adder
        self.num_objects = (
            [num_objects] if isinstance(num_objects, int) else num_objects
        )

    def __len__(self):
        return len(self._fmo_loader)

    def __getitem__(self, index):
        blurs, frames = self._fmo_loader[index]
        n_obj = random.choice(self.num_objects)

        if n_obj > 1:
            indices = [i for i in range(len(self._fmo_loader)) if i != index]
            fmos = [self._fmo_loader[i] for i in random.sample(indices, n_obj - 1)]
            blurs, frames = (
                [alpha_composite(*imgs, mode="RGBA")[0] for imgs in zip(*data)]
                for data in zip((blurs, frames), *fmos)
            )

        imgs, bgs = self._bg_adder(blurs)

        return {
            "imgs": to_tensor_stack(imgs),
            "blurs": to_tensor_stack(blurs),
            "frames": to_tensor_stack(frames),
            "bgs": to_tensor_stack(bgs),
            "n_obj": n_obj,
        }
