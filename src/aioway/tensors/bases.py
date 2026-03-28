# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import operator
import typing
from abc import ABC
from collections.abc import Iterator
from enum import Enum
from enum import auto as Auto

from torch import Tensor
from torch._tensor import Tensor

from aioway import fake
from aioway._previews import Attr
from aioway.fns import Fn

__all__ = ["TensorFnState", "TensorFnState"]


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

    def __invert__(self):
        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.invert, self)

    def __neg__(self):
        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.neg, self)

    def __add__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.add, self, other)

    def __sub__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.sub, self, other)

    def __mul__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.mul, self, other)

    def __truediv__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.truediv, self, other)

    def __floordiv__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.floordiv, self, other)

    def __mod__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.mod, self, other)

    def __pow__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.pow, self, other)

    @typing.no_type_check
    def __eq__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.eq, self, other)

    @typing.no_type_check
    def __ne__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.ne, self, other)

    def __gt__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.gt, self, other)

    def __ge__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.ge, self, other)

    def __lt__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.lt, self, other)

    def __le__(self, other: TensorFn):

        from .thunks import AnyThunkTensorFn

        return AnyThunkTensorFn(operator.le, self, other)

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
