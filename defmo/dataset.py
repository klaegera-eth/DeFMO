import json
import torch
import zipfile
import numpy as np
from PIL import Image, ImageSequence
from torchvision.transforms.functional import to_tensor


class ZipDataset(torch.utils.data.Dataset):
    max_contrast_tries = 10

    def __init__(self, zip, background_loader, item_range=(0, 1), min_contrast=255 / 10):
        self._zip = zipfile.ZipFile(zip)
        self.params = json.loads(self._zip.comment)
        self._bg_loader = background_loader
        self.min_contrast = min_contrast

        start, end = item_range
        if end > 1:
            self.range = range(start, end)
        else:
            length = len(self._zip.filelist)
            self.range = range(int(start * length), int(end * length))

    def __len__(self):
        return len(self.range)

    def __getitem__(self, index):
        with self._zip.open(self._zip.filelist[self.range[index]]) as f:
            seq = ImageSequence.all_frames(Image.open(f))
            n_blurs = len(self.params["blurs"])
            blurs, frames = seq[:n_blurs], seq[n_blurs:]

        imgs, bgs = self._select_bgs(blurs)

        for collection in [imgs, blurs, frames, bgs]:
            for i, x in enumerate(collection):
                collection[i] = to_tensor(x)

        return {
            "imgs": torch.stack(imgs),
            "imgs_short_cat": torch.cat(imgs[1:]),
            "blurs": torch.stack(blurs),
            "frames": torch.stack(frames),
            "bgs": torch.stack(bgs),
        }

    def _select_bgs(self, blurs):
        # find bgs with min_contrast (only first blur checked)
        # use best out of max_contrast_tries if none found
        best_contrast = -1
        for _ in range(self.max_contrast_tries):
            bgs = self._bg_loader.get_random_seq(len(blurs))
            bg_0 = self._bg_loader.load_image(bgs[0])
            imgbg_0 = self._add_bg(blurs[0], bg_0)
            alpha_mask = np.array(blurs[0])[:, :, 3] > 0
            contrast = self._contrast(*imgbg_0, mask=alpha_mask)
            if contrast > best_contrast:
                best_contrast = contrast
                best_imgbg_0 = imgbg_0
                best_bgs = bgs
            if contrast >= self.min_contrast:
                break

        bgs = [self._bg_loader.load_image(bg) for bg in best_bgs[1:]]
        imgbgs = [best_imgbg_0] + [self._add_bg(*ib) for ib in zip(blurs[1:], bgs)]
        return [list(l) for l in zip(*imgbgs)]

    def _add_bg(_, img, bg):
        bg = bg.resize(img.size).convert(img.mode)
        img = Image.alpha_composite(bg, img).convert("RGB")
        return img, bg.convert(img.mode)

    def _contrast(_, img1, img2, mask=None):
        diff = abs(np.array(img1, dtype=int) - img2)
        return np.mean(diff[mask] if mask is not None else diff)
