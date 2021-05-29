import zipfile


class ZipFile(zipfile.ZipFile):
    """Pickleable ZipFile wrapper."""

    def __init__(self, *args, **kwargs):
        self.init_args = args, kwargs
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        return self.init_args

    def __setstate__(self, state):
        args, kwargs = state
        self.__init__(*args, **kwargs)
