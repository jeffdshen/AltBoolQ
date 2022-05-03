import random
import numpy as np
import torch
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.use_deterministic_algorithms(False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True)


def set_environ():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
