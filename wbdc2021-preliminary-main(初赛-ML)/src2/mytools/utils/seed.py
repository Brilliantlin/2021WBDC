import numpy as np
import os
import random
import torch
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_list(seed):
    np.random.seed(seed)
    random_seeds = np.random.randint(0, 10000, size=5)
    print('Gnerate random seeds:', random_seeds)
    return random_seeds
