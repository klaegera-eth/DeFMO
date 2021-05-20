import json
import zipfile
from PIL import Image, ImageSequence


class FmoLoader:
    def __init__(self, zip, item_range=(0, 1)):
        # save args for pickling
        self._args = locals()
        del self._args["self"]

        self._zip = zipfile.ZipFile(zip)
        self.params = json.loads(self._zip.comment)

        start, end = item_range
        if end > 1:
            self.range = range(start, end)
        else:
            length = len(self._zip.filelist)
            self.range = range(int(start * length), int(end * length))

    def __getstate__(self):
        return self._args

    def __setstate__(self, state):
        self.__init__(**state)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, index):
        with self._zip.open(self._zip.filelist[self.range[index]]) as f:
            seq = ImageSequence.all_frames(Image.open(f))
            n_blurs = len(self.params["blurs"])
            blurs, frames = seq[:n_blurs], seq[n_blurs:]
            return blurs, frames
