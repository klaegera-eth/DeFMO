import os
import random
import fnmatch
import tempfile
from zipfile import ZipFile
from collections import defaultdict
from contextlib import contextmanager


class ZipLoader:
    def __init__(self, zip, filter="*[!/]"):
        print("Loading", zip)
        self.zip = ZipFile(zip)
        self.names = fnmatch.filter(self.zip.namelist(), filter)
        self.dirtree = None

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

    def get_random(self, balance_subdirs=False):
        if balance_subdirs:
            if not self.dirtree:
                # create directory tree of zip contents
                dict_tree = lambda: defaultdict(dict_tree)
                self.dirtree = dict_tree()
                for name in self.names:
                    node = self.dirtree
                    for d in name.split("/")[:-1]:
                        node = node[d]
                    node[name] = None
            # randomly sample at every level of directory tree
            node = self.dirtree
            while True:
                name = random.choice(list(node.keys()))
                node = node[name]
                if not node:
                    # leaf node
                    return name
        return random.choice(self.names)
