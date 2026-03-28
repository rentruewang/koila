# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import operator
import typing
from abc import ABC
from collections.abc import Iterator
from typing import Any

from torch import Tensor
from torch._tensor import Tensor

from aioway import fake
from aioway._previews import Attr
from aioway.fn import Fn

__all__ = ["TensorFn"]


class TensorFn(Fn[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        assert fake.is_fake_tensor(self._fake_result)
        self.__attr = Attr.from_tensor(self._fake_result)

    def __invert__(self) -> TensorFn:
        from ._thunks import UFunc1Thunk

        return UFunc1Thunk(operator.invert, self)

    def __neg__(self) -> TensorFn:
        from ._thunks import UFunc1Thunk

        return UFunc1Thunk(operator.neg, self)

    def __add__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.add, self, other)

    def __sub__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.sub, self, other)

    def __mul__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.mul, self, other)

    def __truediv__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.truediv, self, other)

    def __floordiv__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.floordiv, self, other)

    def __mod__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.mod, self, other)

    def __pow__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.pow, self, other)

    @typing.no_type_check
    def __eq__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.eq, self, other)

    @typing.no_type_check
    def __ne__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.ne, self, other)

    def __gt__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.gt, self, other)

    def __ge__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.ge, self, other)

    def __lt__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.lt, self, other)

    def __le__(self, other: Any) -> TensorFn:
        from ._thunks import UFunc2Thunk

        return UFunc2Thunk(operator.le, self, other)

    def attr(self) -> Attr:
        return self.__attr

    @abc.abstractmethod
    @typing.override
    def forward(self) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> Iterator[Fn]:
        """
        Yields the dependent `Fn`s.
        """

        raise NotImplementedError

    @classmethod
    def from_tensor(cls, data: Tensor, /) -> TensorFn:
        from ._data import TensorDataFn

        return TensorDataFn(data)
