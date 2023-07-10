import os
from .logger import Logger

logger = Logger("PROBE")
STORE_DIR = "./probe"

def set_store_dir(dir):
    global STORE_DIR
    STORE_DIR = dir

def get_store_dir(dir=None):
    global STORE_DIR
    if not dir:
        path = STORE_DIR
    else:
        path = os.path.join(STORE_DIR, dir)
    os.makedirs(path, exist_ok=True)
    return path


def probe(local):
    import code
    exec("from exp_tools.probe import *")
    code.interact(local=local)


# UTILITIES
def hist(data):
    from matplotlib import pyplot as plt
    plt.hist(data, bins=25)

def savefig(name, store_dir=None, fig=None, **kwargs):
    from matplotlib import pyplot as plt
    store_dir = get_store_dir(store_dir)
    fname = '.'.join(name.split(".")[:-1])
    title = " ".join([n.capitalize() for n in fname.split("_")])
    if fig:
        ax = fig.axes[0]
        ax.set_title(title)
        fig.savefig(os.path.join(store_dir, name), bbox_inches="tight", **kwargs)
    else:
        plt.title(title)
        plt.savefig(os.path.join(store_dir, name), bbox_inches="tight", **kwargs)
    plt.clf()

prev_time = None
def timeit(name="", start=False):
    import time
    global prev_time
    now = time.time()
    if start:
        prev_time = None
    if not prev_time:
        logger.log(f"Starting at time {now}.")
    else:
        logger.log(f"{'Time' if not name else name} elapsed: {now-prev_time:.4f} sec.")
    prev_time = now

def select(data, indices):
    if isinstance(data, list):
        return [data[s] for s in indices]
    else:
        return data[indices]

def linspace_select(data, num=3):
    import numpy as np
    indices = np.linspace(0, len(data)-1, num)
    indices = [int(i) for i in indices]
    return indices

def random_select(data, num=3):
    import numpy as np
    N = len(data)
    indices = np.random.permutation(N)[:num]
    return indices

def split_select(data, portion=.5):
    import numpy as np
    N = len(data)
    num = int(N*portion)
    ids = np.random.permutation(N)
    half1 = ids[:num]
    half2 = ids[num:]
    return half1, half2