# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import operator
import typing
from abc import ABC
from collections.abc import Iterator
from enum import Enum
from enum import auto as Auto
from typing import Any

from torch import Tensor
from torch._tensor import Tensor

from aioway import fake
from aioway._previews import Attr
from aioway.fn import Fn

__all__ = ["TensorFnState", "TensorFn"]


class TensorFnState(Enum):
    "The status of a `Later` object."

    PENDING = Auto()
    "The object is pending evaluation."

    EVALUATED = Auto()
    "The object is evaluated."


class TensorFn(Fn[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()

        self.__real_result: Tensor | None = None

        with fake.enable():
            self.__fake_result: Tensor = self.forward()

        assert fake.is_fake_tensor(self.__fake_result)
        self.__attr = Attr.from_tensor(self.__fake_result)

    def __invert__(self) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.invert, self)

    def __neg__(self) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.neg, self)

    def __add__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.add, self, other)

    def __sub__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.sub, self, other)

    def __mul__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.mul, self, other)

    def __truediv__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.truediv, self, other)

    def __floordiv__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.floordiv, self, other)

    def __mod__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.mod, self, other)

    def __pow__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.pow, self, other)

    @typing.no_type_check
    def __eq__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.eq, self, other)

    @typing.no_type_check
    def __ne__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.ne, self, other)

    def __gt__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.gt, self, other)

    def __ge__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.ge, self, other)

    def __lt__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.lt, self, other)

    def __le__(self, other: Any) -> TensorFn:
        from . import _thunks

        return _thunks.thunk(operator.le, self, other)

    def attr(self) -> Attr:
        return self.__attr

    @typing.override
    def do(self) -> Tensor:
        """
        Perform the computation.

        If the result is previously stored, use the stored result.
        If we are in fake mode, return the `FakeTensor` version.

        Returns:
            A `FakeTensor` if in fake mode, a `Tensor` otherwise.
        """

        if fake.detect_fake_mode():
            return self.__fake_result

        if self.__real_result is None:
            self.__real_result = self.forward()

        assert fake.is_real_tensor(self.__real_result)
        return self.__real_result

    @abc.abstractmethod
    def forward(self) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> Iterator[Fn]:
        """
        Yields the dependent `Fn`s.
        """

        raise NotImplementedError

    @property
    def state(self) -> TensorFnState:
        if self.__real_result is None:
            return TensorFnState.PENDING
        else:
            return TensorFnState.EVALUATED

    @classmethod
    def from_tensor(cls, data: Tensor, /) -> TensorFn:
        from ._data import TensorDataFn

        return TensorDataFn(data)
