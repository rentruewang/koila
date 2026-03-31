# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import tensordict as td
import torch
from torch import cuda

from aioway.chunks import Chunk
from aioway.tdicts import AttrSet
from aioway.tensors import Attr


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


def chunk_ok(*, size: int, device: str) -> Chunk:
    data = td.TensorDict(
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
        f1d=Attr.parse(
            device="cpu",
            shape=[1],
            dtype="float32",
        ),
        f2d=Attr.parse(
            device="cpu",
            shape=[1, 32],
            dtype="float32",
        ),
        i1d=Attr.parse(
            device="cpu",
            shape=[1],
            dtype="int64",
        ),
        i2d=Attr.parse(
            device="cpu",
            shape=[1, 32],
            dtype="int64",
        ),
    )
    return Chunk.from_data_schema(data=data, schema=schema)


def unionable_ok(*, size: int, device: str):
    return chunk_ok(size=size + 1, device=device)


def concat_ok(*, size: int, device: str):
    return chunk_ok(size=size, device=device)


def random_things():
    return [1, None, "hello", torch.tensor([100]), np.arange(100).reshape(4, 25)]
