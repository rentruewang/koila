# Copyright (c) AIoWay Authors - All Rights Reserved


import numpy as np
import torch
from tensordict import TensorDict
from torch import cuda

from aioway.ops import _funcs


def cpu_and_maybe_cuda():
    """
    The devices used in the tests.

    """

    devs = ["cpu"]

    if cuda.is_available():
        devs.append("cuda")

    return devs


def batch_sizes():
    yield 16
    yield 64
    yield 1024


def tensordict_ok(*, size: int, device: str):
    return TensorDict(
        {
            "f1d": torch.randn(size),
            "f2d": torch.randn(size, 32),
            "i1d": torch.randint(0, 100, [size]),
            "i2d": torch.randint(0, 100, [size, 32]),
        },
        batch_size=size,
        device=device,
    )


def tensordict_no_batch(*, size: int, device: str):
    return TensorDict(
        {"f1d": torch.randn(size), "f2d": torch.randn(size, 32)},
        device=device,
    )


def unionable_ok(*, size: int, device: str):
    return tensordict_ok(size=size + 1, device=device)


def concat_ok(*, size: int, device: str):
    td = tensordict_ok(size=size, device=device)
    return _funcs.rename(td, f1d="f1", f2d="f2", i1d="i1", i2d="i2")


def random_things():
    return [1, None, "hello", torch.tensor([100]), np.arange(100).reshape(4, 25)]
