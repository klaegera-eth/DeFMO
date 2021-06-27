import json
from PIL import Image, ImageSequence

from . import ZipFile


class FmoLoader:
    def __init__(self, zip, item_range=(0, 1), blurs=None, dummy_frames=0):
        self._zip = ZipFile(zip)
        self.params = json.loads(self._zip.comment)

        self.blurs = blurs
        self.dummy_frames = dummy_frames

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
            if self.blurs:
                blurs = [blurs[i] for i in self.blurs]
            frames += [Image.new("RGBA", blurs[0].size)] * self.dummy_frames
            return blurs, frames
