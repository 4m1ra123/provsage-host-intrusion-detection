"""Utilitaire de déterminisme — version assouplie pour PyG + MIG A100.

Note : torch.use_deterministic_algorithms(True) est désactivé car il
provoque des erreurs NVML avec les opérations scatter de PyG sur les
slices MIG. La reproductibilité reste forte grâce aux seeds fixés et
à cudnn.deterministic = True.
"""
import os, random
import numpy as np
import torch


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[DETERMINISM] seed={seed}, cudnn.deterministic=True, seeds fixed")
