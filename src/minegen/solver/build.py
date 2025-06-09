import torch
from typing import Dict, Any

from ..config.models import Config


def make_optimiser(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer from config"""
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.solver.base_lr
        weight_decay = cfg.solver.weight_decay
        if "bias" in key:
            lr = cfg.solver.base_lr * cfg.solver.bias_lr_factor
            weight_decay = cfg.solver.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    optimizer_class = getattr(torch.optim, cfg.solver.optimizer_name)
    optimiser = optimizer_class(params, momentum=cfg.solver.momentum)
    return optimiser