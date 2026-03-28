# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import Callable, Iterator
from enum import Enum
from enum import auto as Auto
from typing import Any

from torch import Tensor
from torch._tensor import Tensor

from aioway import _common, fake
from aioway._previews import Attr
from aioway.fns import Fn

__all__ = ["TensorDataFn"]


class TensorFnState(Enum):
    "The status of a `Later` object."

    PENDING = Auto()
    "The object is pending evaluation."

    EVALUATED = Auto()
    "The object is evaluated."


class TensorFn(Fn[Tensor], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.__state = TensorFnState.PENDING

        self.__result: Tensor | None = None

        with fake.enable():
            tensor = self.do()

        assert fake.is_fake_tensor(tensor)
        self.__attr = Attr.from_tensor(tensor)

    def attr(self) -> Attr:
        return self.__attr

    @typing.override
    def do(self) -> Tensor:
        """
        Perform the computation.

        If the result is previously stored, use the stored result.
        If we are in fake mode, do not store the result.

        Returns:
            A `FakeTensor` if in fake mode, a `Tensor` otherwise.
        """

        if self.__result is not None:
            assert fake.is_real_tensor(self.__result)
            return self.__result

        result = self.forward()

        # Only store the tensor if we're not in fake mode!
        if not fake.detect_fake_mode():
            self.__result = result

        return result

    @abc.abstractmethod
    def forward(self) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    @typing.override
    def _deps(self) -> Iterator[Fn]:
        raise NotImplementedError

    @property
    def state(self) -> TensorFnState:
        if self.__result is None:
            return TensorFnState.PENDING
        else:
            return TensorFnState.EVALUATED


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


class ThunkTensorFn(TensorFn):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    def __init__(
        self, func: Callable[..., Tensor], *args: Fn[Any], **kwargs: Fn[Any]
    ) -> None:
        super().__init__()

        self._func = func

        for arg in self.args:
            if not isinstance(arg, Fn):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, Fn):
                raise TypeError(f"{key}={value} is not a `Thunk`")

        self._args = args
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def forward(self) -> Tensor:
        args = [arg.do() for arg in self.args]
        kwargs = {key: val.do() for key, val in self.kwargs.items()}
        return self.func(*args, **kwargs)

    @typing.override
    def _deps(self) -> Iterator[Fn[object]]:
        yield from self.args
        yield from self.kwargs.values()

    @property
    def func(self) -> Callable[..., Tensor]:
        return self._func

    @property
    def args(self) -> tuple[Fn[Any], ...]:
        return self._args

    @property
    def kwargs(self) -> dict[str, Fn[Any]]:
        return self._kwargs
