import os
import random
import fnmatch
import tempfile
from zipfile import ZipFile
from collections import defaultdict
from contextlib import contextmanager


class ZipLoader:
    def __init__(self, zip, filter="*[!/]", balance_subdirs=False):
        self.zip = ZipFile(zip)
        self.names = fnmatch.filter(self.zip.namelist(), filter)
        self.tree = None
        if balance_subdirs:
            # create directory tree of zip contents
            dict_tree = lambda: defaultdict(dict_tree)
            self.tree = dict_tree()
            for name in self.names:
                node = self.tree
                for d in name.split("/")[:-1]:
                    node = node[d]
                node[name] = None

    @contextmanager
    def get_path(self, name):
        _, ext = os.path.splitext(name)
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(self.zip.read(name))
        try:
            yield path
        finally:
            os.remove(path)

    def get_random_path(self):
        if self.tree:
            # randomly sample at every level of directory tree
            node = self.tree
            while True:
                name = random.choice(list(node.keys()))
                node = node[name]
                if not node:
                    # leaf node
                    return self.get_path(name)
        return self.get_path(random.choice(self.names))
