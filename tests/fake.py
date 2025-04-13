# Copyright (c) RenChu Wang - All Rights Reserved


import numpy as np
import torch
from tensordict import TensorDict
from torch import cuda

from aioway.blocks import Block


def cpu_and_maybe_cuda():
    """
    The devices used in the tests.

    todo))
        Add multi-gpu support.

    todo))
        Test distributed.
    """

    devs = ["cpu"]

    if cuda.is_available():
        devs.append("cuda")

    return devs


def block_sizes():
    for i in range(0, 11, 2):
        yield 2**i


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


def block_ok(*, size: int, device: str):
    tensordict = tensordict_ok(device=device, size=size)
    return Block(tensordict)


def unionable_block_ok(*, size: int, device: str):
    return block_ok(size=size + 1, device=device)


def concat_block_ok(*, size: int, device: str):
    return block_ok(size=size, device=device).rename(
        f1d="f1", f2d="f2", i1d="i1", i2d="i2"
    )


def random_things():
    return [1, None, "hello", torch.tensor([100]), np.arange(100).reshape(4, 25)]
