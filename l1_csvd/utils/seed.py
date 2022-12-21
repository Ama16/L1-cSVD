import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Set random seed and cudnn params.

    Args:
        seed: Random state.
        deterministic: cudnn backend.

    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
