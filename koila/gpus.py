from __future__ import annotations

from pynvml.smi import nvidia_smi
from torch import cuda

from . import constants

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
    gpu = query["gpu"][0]

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
    stats = cuda.memory_stats(0)

    reserved_memory = stats["reserved_bytes.all.current"]
    active_memory = stats["active_bytes.all.current"]
    unused_memory = stats["inactive_split_bytes.all.current"]

    assert unused_memory == reserved_memory - active_memory
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
