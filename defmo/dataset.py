import json
import torch
import zipfile
from PIL import Image, ImageSequence
from torchvision.transforms.functional import to_tensor


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip, background_loader):
        self.zip = zipfile.ZipFile(zip)
        self.params = json.loads(self.zip.comment)
        self.bg_loader = background_loader

    def __len__(self):
        return len(self.zip.filelist)

    def __getitem__(self, index):
        img = Image.open(self.zip.open(self.zip.filelist[index]))
        frames = ImageSequence.all_frames(img)
        n_blurs = len(self.params["blurs"])
        blurs, frames = frames[:n_blurs], frames[n_blurs:]
        bgs = self.bg_loader.get_random_seq(n_blurs)
        for i, (blur, bg) in enumerate(zip(blurs, bgs)):
            bg = Image.open(self.bg_loader.zip.open(bg))
            bg = bg.resize(blur.size).convert(blur.mode)
            blurs[i] = Image.alpha_composite(bg, blur).convert("RGB")
        return {
            "blurs": torch.cat([to_tensor(blur) for blur in blurs]),
            "frames": torch.stack([to_tensor(frame) for frame in frames]),
        }
