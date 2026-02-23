from __future__ import annotations

import random

import numpy as np
import torch


def set_global_determinism(seed: int, deterministic: bool = True) -> None:
    """Set deterministic seeds and backend flags for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # use_deterministic_algorithms can throw if an op has no deterministic path;
        # warn_only keeps runs reproducible where possible without hard-failing.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
