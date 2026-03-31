# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Iterator

import torch

from aioway import _common, fake
from aioway.fn import Fn

from .fn import TensorFn

__all__ = ["TensorDataFn"]


@_common.dcls_no_eq
class TensorDataFn(TensorFn):
    "The `Fn` representing a plain `torch.Tensor`."

    data: torch.Tensor

    def __post_init__(self) -> None:
        super().__init__()

        # Mark as `EVALUATED`.
        _ = self.do()

    @typing.override
    def forward(self):
        if mode := fake.is_enabled():
            converter = mode.fake_tensor_converter
            return converter.from_real_tensor(mode, self.data)

        else:
            return self.data

    @typing.override
    def _deps(self) -> Iterator[Fn[torch.Tensor]]:
        return
        yield
