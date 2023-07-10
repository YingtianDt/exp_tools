import os
import glob
from .core import Node, Executor

INACTIVE_PREFIX = "_"

def from_directory(directory, *args, **kwargs):
    fpaths = glob.glob(os.path.join(directory, "**/*.py"), recursive=True)
    nodes = [Node.from_script(fpath) for fpath in fpaths if not os.path.basename(fpath).startswith(INACTIVE_PREFIX)]
    executor = Executor(nodes, *args, **kwargs)
    return executor