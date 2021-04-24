import sys


def print_stderr(*args, **kwargs):
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    print(*args, **kwargs)
    sys.stderr.flush()
