import os
import sys
import random
import fnmatch
import tempfile
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
        self.zip = ZipFile(zip)
        self.names = fnmatch.filter(self.zip.namelist(), filter)
        self.dirtree = None

        if balance_subdirs:
            # create directory tree of zip contents
            dict_tree = lambda: defaultdict(dict_tree)
            self.dirtree = dict_tree()
            for name in self.names:
                node = self.dirtree
                for d in name.split("/")[:-1]:
                    node = node[d]
                node[name] = None

    @contextmanager
    def as_tempfile(self, name):
        _, ext = os.path.splitext(name)
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(self.zip.read(name))
        try:
            yield path
        finally:
            os.remove(path)

    def get_random(self):
        if self.dirtree:
            # randomly sample at every level of directory tree
            node = self.dirtree
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
            node = self.dirtree
            for d in seed.split("/")[:-1]:
                node = node[d]
            names = sorted(node.keys())
            if len(names) >= length:
                start = random.randint(0, len(names) - length)
                return names[start : start + length]
        raise ValueError(f"Failed to get random sequence of length {length}.")
