import numpy as np
from PIL import Image

from .utils import alpha_composite, mean_diff


class BackgroundAdder:
    def __init__(
        self,
        background_loader,
        min_contrast=255 / 10,
        max_contrast_tries=10,
    ):
        self.background_loader = background_loader
        self.min_contrast = min_contrast
        self.max_contrast_tries = max_contrast_tries

    def __call__(self, inputs):

        # find bgs with min_contrast (only first blur checked)
        # use best out of max_contrast_tries if none found
        best_contrast = -1
        for _ in range(self.max_contrast_tries):
            bg_indices = self.background_loader.get_random_seq(len(inputs))

            # contrast only checked on first blur in sequence
            fmo, bg = alpha_composite(
                self.background_loader.load_image(bg_indices[0]),
                inputs[0],
            )

            contrast = mean_diff(fmo, bg, mask=np.array(inputs[0])[:, :, 3] > 0)

            if contrast > best_contrast:
                best_contrast = contrast

                # also save result of first blur to avoid recomputing
                best_first = fmo, bg
                best_bg_indices = bg_indices

            if contrast >= self.min_contrast:
                break

        fmo_bg_pairs = [best_first] + [
            alpha_composite(
                self.background_loader.load_image(bg_index),
                blur,
            )
            for blur, bg_index in zip(inputs[1:], best_bg_indices[1:])
        ]

        return [list(l) for l in zip(*fmo_bg_pairs)]

    class Constant:
        def __init__(self, color="white"):
            self.bg = Image.new("RGB", (1, 1), color)

        def __call__(self, inputs):
            fmo_bg_pairs = [alpha_composite(self.bg, input) for input in inputs]
            return [list(l) for l in zip(*fmo_bg_pairs)]
