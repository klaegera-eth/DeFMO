import numpy as np
from PIL import Image


from torchvision.transforms.functional import to_tensor, to_pil_image


def alpha_composite(bg, *layers, mode="RGB"):
    size = layers[0].size
    *layers, bg = [img.resize(size).convert("RGBA") for img in (*layers, bg)]
    img = bg
    for layer in layers:
        img = Image.alpha_composite(img, layer)
    return img.convert(mode), bg.convert(mode)


def mean_diff(img1, img2, mask=None):
    diff = abs(np.array(img1, dtype=int) - img2)
    return np.mean(diff[mask] if mask is not None else diff)
