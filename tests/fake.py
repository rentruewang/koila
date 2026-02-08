# Copyright (c) AIoWay Authors - All Rights Reserved


import numpy as np
import torch
from tensordict import TensorDict
from torch import cuda

from aioway import attrs
from aioway.attrs import AttrSet
from aioway.batches import Chunk


def cpu_and_maybe_cuda() -> list[str]:
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


def chunk_ok(*, size: int, device: str):
    data = TensorDict(
        {
            "f1d": torch.randn(size),
            "f2d": torch.randn(size, 32),
            "i1d": torch.randint(0, 100, [size]),
            "i2d": torch.randint(0, 100, [size, 32]),
        },
        batch_size=size,
        device=device,
    )
    schema = AttrSet.from_values(
        f1d=attrs.attr(
            device="cpu",
            shape=(),
            dtype="float32",
        ),
        f2d=attrs.attr(
            device="cpu",
            shape=32,
            dtype="float32",
        ),
        i1d=attrs.attr(
            device="cpu",
            shape=(),
            dtype="int64",
        ),
        i2d=attrs.attr(
            device="cpu",
            shape=32,
            dtype="int64",
        ),
    )
    return Chunk(data=data, schema=schema)


def unionable_ok(*, size: int, device: str):
    return chunk_ok(size=size + 1, device=device)


def concat_ok(*, size: int, device: str):
    return chunk_ok(size=size, device=device)


def random_things():
    return [1, None, "hello", torch.tensor([100]), np.arange(100).reshape(4, 25)]
