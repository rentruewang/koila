# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import operator
import typing
from abc import ABC
from collections.abc import Iterator
from enum import Enum
from enum import auto as Auto
from typing import Any, ClassVar

from torch import Tensor
from torch._tensor import Tensor

from aioway import fake
from aioway._previews import Attr
from aioway.fn import Fn, FnState

__all__ = ["Fn", "FnState", "TensorFn"]


class FnState(Enum):
    "The status of a `Later` object."

    PENDING = Auto()
    "The object is pending evaluation."

    EVALUATED = Auto()
    "The object is evaluated."


class Fn[T](ABC):
    """
    `Fn`s represent computation that shall be done later.

    Like Haskell's thunks, once evaluated,
    the value is stored in the `Fn` itself and never re-evaluated.
    The value shall be gone during GC.

    I was going to go for `Op` but it's used a lot in `torch`.
    """

    __match_args__: ClassVar[tuple[str, ...]]

    def __init__(self) -> None:
        super().__init__()

        self._real_result: T | None = None

        with fake.enable():
            self._fake_result: T = self.forward()

        assert fake.is_fake_tensor(self._fake_result)

    @abc.abstractmethod
    def do(self) -> T:
        """
        Do the computation.
        """

        raise NotImplementedError

    @functools.cached_property
    def deps(self):
        "The depedent `Exec`s."

        return tuple(self._deps())

    @property
    def is_leaf(self) -> bool:
        "Whether or not the thunk is dependent on other thunks. If not, it's a leaf."
        return not self.deps

    @abc.abstractmethod
    def _deps(self) -> Iterator[Fn[Any]]:
        """
        Return the depedent thunks.
        """

    @abc.abstractmethod
    def forward(self) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def state(self) -> FnState:
        raise NotImplementedError


class TensorFn(Fn[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()

        self._real_result: Tensor | None = None

        with fake.enable():
            self._fake_result: Tensor = self.forward()

        assert fake.is_fake_tensor(self._fake_result)
        self.__attr = Attr.from_tensor(self._fake_result)

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
            return self._fake_result

        if self._real_result is None:
            self._real_result = self.forward()

        assert fake.is_real_tensor(self._real_result)
        return self._real_result

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
    def state(self) -> FnState:
        if self._real_result is None:
            return FnState.PENDING
        else:
            return FnState.EVALUATED

    @classmethod
    def from_tensor(cls, data: Tensor, /) -> TensorFn:
        from ._data import TensorDataFn

        return TensorDataFn(data)
