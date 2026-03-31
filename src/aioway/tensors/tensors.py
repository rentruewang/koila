# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import operator
import typing
from abc import ABC
from collections import abc as cabc

import torch

from aioway import fake, fn

from . import attrs

__all__ = ["TensorFn", "tensor"]


class TensorFn(fn.Fn[torch.Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        assert fake.is_fake_tensor(self._fake_result), type(self._fake_result)
        self.__attr = attrs.Attr.from_tensor(self._fake_result)

    def __len__(self) -> int:
        return self.attr.shape[0]

    def __getitem__(self, key: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.GatherThunk(self, key)

    def __invert__(self) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc1Thunk(operator.invert, self)

    def __neg__(self) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc1Thunk(operator.neg, self)

    def __add__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.add, self, other)

    def __sub__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.sub, self, other)

    def __mul__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.mul, self, other)

    def __truediv__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.truediv, self, other)

    def __floordiv__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.floordiv, self, other)

    def __mod__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.mod, self, other)

    def __pow__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.pow, self, other)

    @typing.no_type_check
    def __eq__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.eq, self, other)

    @typing.no_type_check
    def __ne__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.ne, self, other)

    def __gt__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.gt, self, other)

    def __ge__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.ge, self, other)

    def __lt__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.lt, self, other)

    def __le__(self, other: typing.Any) -> TensorFn:
        from . import _thunks

        return _thunks.UFunc2Thunk(operator.le, self, other)

    @property
    def attr(self) -> attrs.Attr:
        return self.__attr

    @abc.abstractmethod
    @typing.override
    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> cabc.Iterator[fn.Fn]:
        """
        Yields the dependent `fn.Fn`s.
        """

        raise NotImplementedError

    @classmethod
    def from_tensor(cls, data: torch.Tensor, /) -> TensorFn:
        from ._data import TensorDataFn

        return TensorDataFn(data)


def tensor(data: TensorFn | torch.Tensor) -> TensorFn:

    if isinstance(data, TensorFn):
        return data

    if isinstance(data, torch.Tensor):
        return TensorFn.from_tensor(data)

    raise TypeError(f"Do not know how to handle {type(data)=}.")
