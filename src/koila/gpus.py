# Copyright (c) RenChu Wang - All Rights Reserved

from __future__ import annotations

import math
from typing import Generator

from pynvml.smi import nvidia_smi
from torch import cuda

from . import constants
from .interfaces import BatchedPair

NVSMI = None


def nvidia_free_memory() -> int:
    """
    Calls nvidia's nvml library and queries available GPU memory.
    Currently the function only works with 1 GPU.

    Returns
    -------

    Free GPU memory in terms of bytes.
    """

    global NVSMI
    if NVSMI is None:
        NVSMI = nvidia_smi.getInstance()

    assert NVSMI is not None
    query = NVSMI.DeviceQuery("memory.free")

    # Only works on one GPU as of now.
    gpu = query["gpu"][0]["fb_memory_usage"]

    unit = constants.UNITS[gpu["unit"].lower()]
    free = gpu["free"]

    return free * unit


def torch_free_memory() -> int:
    """
    Calls torch's memory statistics to calculate the amount of GPU memory unused.
    Currently the function only works with 1 GPU.

    Returns
    -------

    Reserved GPU memory in terms of bytes.
    """

    if not cuda.is_available():
        return 0

    # Only works on one GPU as of now.

    reserved_memory = cuda.memory_reserved(0)
    active_memory = cuda.memory_allocated(0)
    unused_memory = reserved_memory - active_memory
    return unused_memory


def free_memory() -> int | None:
    """
    The amount of free GPU memory that can be used.

    Returns
    -------

    Unused GPU memory, or None if no GPUs are available.
    """

    if cuda.is_available():
        return nvidia_free_memory() + torch_free_memory()
    else:
        return None


def maximum_batch(memory: BatchedPair, total_memory: int | None = None) -> int | None:
    # batch * x + no_batch = unused_memoroy
    if total_memory is None:
        total_memory = free_memory()

    if total_memory is None:
        return None

    return (total_memory - memory.no_batch) // memory.batch


def split_batch(
    memory: BatchedPair, current_batch: int, total_memory: int | None = None
) -> Generator[int, None, None]:
    max_batch = maximum_batch(memory, total_memory)

    if max_batch is None:
        yield current_batch
        return

    batch_size = 2 ** (math.floor(math.log2(max_batch)))
    (times, current_batch) = divmod(current_batch, batch_size)

    for _ in range(times):
        yield batch_size

    while current_batch > 0:
        batch_size >>= 1
        if current_batch >= batch_size:
            current_batch -= batch_size
            yield batch_size
        assert current_batch < batch_size, [current_batch, batch_size]
