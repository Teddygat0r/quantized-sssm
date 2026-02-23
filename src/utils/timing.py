from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import torch


def maybe_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


@contextmanager
def timed_block(device: torch.device) -> Generator[list[float], None, None]:
    holder: list[float] = []
    maybe_cuda_sync(device)
    start = time.perf_counter()
    yield holder
    maybe_cuda_sync(device)
    end = time.perf_counter()
    holder.append(end - start)
