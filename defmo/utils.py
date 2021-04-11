import os
import sys
import random
import fnmatch
import tempfile
from PIL import Image
from zipfile import ZipFile
from collections import defaultdict
from contextlib import contextmanager


def print_stderr(*args, **kwargs):
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    print(*args, **kwargs)
    sys.stderr.flush()


class ZipLoader:
    def __init__(self, zip, filter="*[!/]", balance_subdirs=False):
        self._zip = ZipFile(zip)
        self.names = fnmatch.filter(self._zip.namelist(), filter)
        self._dirtree = None

        if balance_subdirs:
            # create directory tree of zip contents
            dict_tree = lambda: defaultdict(dict_tree)
            self._dirtree = dict_tree()
            for name in self.names:
                node = self._dirtree
                for d in name.split("/")[:-1]:
                    node = node[d]
                node[name] = None

    def __len__(self):
        return len(self.names)

    def open(self, name):
        return self._zip.open(name)

    def load_image(self, name):
        with self.open(name) as f:
            img = Image.open(f)
            img.load()
            return img

    @contextmanager
    def as_tempfile(self, name):
        _, ext = os.path.splitext(name)
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(self._zip.read(name))
        try:
            yield path
        finally:
            os.remove(path)

    def get_random(self):
        if self._dirtree:
            # randomly sample at every level of directory tree
            node = self._dirtree
            while True:
                name = random.choice(list(node.keys()))
                node = node[name]
                if not node:
                    # leaf node
                    return name
        return random.choice(self.names)

    def get_random_seq(self, length):
        for _ in range(10000):
            seed = self.get_random()
            node = self._dirtree
            for d in seed.split("/")[:-1]:
                node = node[d]
            names = sorted(node.keys())
            if len(names) >= length:
                start = random.randint(0, len(names) - length)
                return names[start : start + length]
        raise ValueError(f"Failed to get random sequence of length {length}.")
