# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections.abc import Iterator

from torch import Tensor
from torch._tensor import Tensor

from aioway import fake
from aioway.fn import Fn

from .fn import TensorFn

__all__ = ["TensorDataFn"]


class TensorDataFn(TensorFn):
    "The `Fn` representing a plain `Tensor`."

    __match_args__ = ("data",)

    def __init__(self, data: Tensor) -> None:
        super().__init__()
        self._data = data
        self.do()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data!r})"

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

    @property
    def data(self) -> Tensor:
        return self._data
