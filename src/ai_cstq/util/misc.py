"""Miscellaneous utilities for BSGM-CellTrack."""

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch import Tensor


class MetricLogger:
    """Rolling average metric logger."""

    def __init__(self, delimiter: str = "  "):
        self.meters: Dict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        parts = [f"{k}: {v}" for k, v in self.meters.items()]
        return self.delimiter.join(parts)

    def log_every(self, iterable, print_freq: int = 10, header: str = ""):
        i = 0
        start = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                elapsed = time.time() - start
                print(f"{header} [{i}]  {self}  time/iter: {elapsed/i:.3f}s")


class SmoothedValue:
    def __init__(self, window: int = 20):
        self.values = []
        self.window = window
        self.total = 0.0
        self.count = 0

    def update(self, v: float):
        self.values.append(v)
        if len(self.values) > self.window:
            self.values.pop(0)
        self.total += v
        self.count += 1

    @property
    def avg(self) -> float:
        return sum(self.values) / max(len(self.values), 1)

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    def __str__(self):
        return f"{self.avg:.4f} ({self.global_avg:.4f})"


def nested_dict_to_device(d: dict, device: torch.device) -> dict:
    """Recursively move tensors in a nested dict to device."""
    out = {}
    for k, v in d.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = nested_dict_to_device(v, device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if isinstance(x, Tensor) else x for x in v]
        else:
            out[k] = v
    return out


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if missing:
        print(f"Missing keys: {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}")
    epoch = ckpt.get("epoch", 0)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return epoch


def get_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
    return total.item()
