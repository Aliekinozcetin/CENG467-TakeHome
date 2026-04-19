import os
import random

import numpy as np
import torch
from transformers import set_seed as transformers_set_seed

SEED = 42

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)
