import json
import torch
import zipfile
from PIL import Image, ImageSequence


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip, background_loader=None):
        self.zip = zipfile.ZipFile(zip)
        self.params = json.loads(self.zip.comment)
        self.bgs = background_loader

    def __len__(self):
        return len(self.zip.filelist)

    def __getitem__(self, index):
        img = Image.open(self.zip.open(self.zip.filelist[index]))
        frames = ImageSequence.all_frames(img)
        n_blurs = len(self.params["blurs"])
        blurs, frames = frames[:n_blurs], frames[n_blurs:]
        if self.bgs:
            bgs = self.bgs.get_random_seq(n_blurs)
            for i, (blur, bg) in enumerate(zip(blurs, bgs)):
                bg = Image.open(self.bgs.zip.open(bg))
                blurs[i] = Image.alpha_composite(bg.resize(blur.size).convert(blur.mode), blur)
        return blurs, frames
