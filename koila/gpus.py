from torch import cuda

NVSMI = None
UNITS = {
    "b": 1,
    "kb": 10 ** 3,
    "kib": 2 ** 10,
    "mb": 10 ** 6,
    "mib": 2 ** 20,
    "gb": 10 ** 9,
    "gib": 2 ** 30,
    "tb": 10 ** 4,
    "tib": 2 ** 40,
}


def nvidia_free_memory() -> int:
    """
    Calls nvidia's nvml library and queries available GPU memory.
    Currently the function only works with 1 GPU.

    Returns
    -------

    Free GPU memory in terms of bytes.
    """

    global NVSMI
    if NVSMI is None and cuda.is_available():
        from pynvml.smi import nvidia_smi

        NVSMI = nvidia_smi.getInstance()
        return 0

    assert NVSMI is not None
    query = NVSMI.DeviceQuery("memory.free")

    # Only works on one GPU as of now.
    gpu = query["gpu"][0]

    unit = UNITS[gpu["unit"].lower()]
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


def free_memory() -> int:
    """
    The amount of free GPU memory that can be used.

    Returns
    -------

    Unused GPU memory.
    """

    return nvidia_free_memory() + torch_free_memory()
