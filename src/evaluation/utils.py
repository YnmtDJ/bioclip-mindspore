import json
import random
import os
import numpy as np
import mindspore


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        return None


def random_seed(seed=42, rank=0):
    mindspore.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


def init_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    if args.gpu != -1:
        device = "0"
        mindspore.set_device("GPU", 0)
    else:
        device = "CPU"
        mindspore.set_device("CPU")
    args.device = device
    device = mindspore.get_context("device_target")
    return device
