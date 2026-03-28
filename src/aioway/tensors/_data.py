# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator

from torch import Tensor
from torch._tensor import Tensor

from aioway import fake
from aioway.fn import Fn

from .fn import TensorFn

__all__ = ["TensorDataFn"]


@dcls.dataclass
class TensorDataFn(TensorFn):
    "The `Fn` representing a plain `Tensor`."

    data: Tensor

    def __post_init__(self) -> None:
        super().__init__()

        # Mark as `EVALUATED`.
        _ = self.do()

    @typing.override
    def forward(self):
        if mode := fake.detect_fake_mode():
            converter = mode.fake_tensor_converter
            return converter.from_real_tensor(mode, self.data)

        else:
            return self.data

    @typing.override
    def _deps(self) -> Iterator[Fn[Tensor]]:
        return
        yield
