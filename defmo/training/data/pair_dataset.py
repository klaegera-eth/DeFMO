import torch
import numpy as np

from .utils import alpha_composite, mean_diff, to_tensor


class PairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        fmo_loader,
        background_loader,
        min_contrast=255 / 10,
        max_contrast_tries=10,
    ):
        self._fmo_loader = fmo_loader
        self._bg_loader = background_loader
        self.min_contrast = min_contrast
        self.max_contrast_tries = max_contrast_tries

    def __len__(self):
        return len(self._fmo_loader)

    def __getitem__(self, index):
        blurs, frames = self._fmo_loader[index]
        imgs, bgs = self._select_bgs(blurs)

        for collection in [imgs, blurs, frames, bgs]:
            for i, x in enumerate(collection):
                collection[i] = to_tensor(x)

        return {
            "imgs": torch.stack(imgs),
            "blurs": torch.stack(blurs),
            "frames": torch.stack(frames),
            "bgs": torch.stack(bgs),
        }

    def _select_bgs(self, blurs):
        # find bgs with min_contrast (only first blur checked)
        # use best out of max_contrast_tries if none found
        best_contrast = -1
        for _ in range(self.max_contrast_tries):
            bg_indices = self._bg_loader.get_random_seq(len(blurs))

            # contrast only checked on first blur in sequence
            fmo, bg = alpha_composite(
                self._bg_loader.load_image(bg_indices[0]),
                blurs[0],
            )

            contrast = mean_diff(fmo, bg, mask=np.array(blurs[0])[:, :, 3] > 0)

            if contrast > best_contrast:
                best_contrast = contrast

                # also save result of first blur to avoid recomputing
                best_first = fmo, bg
                best_bg_indices = bg_indices

            if contrast >= self.min_contrast:
                break

        fmo_bg_pairs = [best_first] + [
            alpha_composite(
                self._bg_loader.load_image(bg_index),
                blur,
            )
            for blur, bg_index in zip(blurs[1:], best_bg_indices[1:])
        ]

        return [list(l) for l in zip(*fmo_bg_pairs)]
