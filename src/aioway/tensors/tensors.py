# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import operator
import typing

import torch

from aioway import fake, fn

from . import attrs

__all__ = ["TensorFn", "tensor", "BasicPreviewFn"]


class TensorFn(fn.Fn[torch.Tensor], abc.ABC):
    """
    `TensorFn` is the `Fn` that would produce a `Tensor`.

    As `Fn` represents lazy computation, the base class acts as a router,
    routing different methods to the corresponding subclasses.

    Subclasses should overwrite these functions:

    1. `do()`: Evaluate and generate the `Tensor`.
    2. `deps()`: The dependent `Fn`, that will be evaluated during `do()`.
    3. `attr()`: The (eager) description of the tensor computation.
    """

    def __len__(self) -> int:
        return self.attr.max_shape[0]

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

    @abc.abstractmethod
    def preview(self) -> attrs.Attr:
        """
        The `preview` function generates a "preview" for the `Tensor` that would be generated.

        The data type (`Attr`) is used to describe the meta data that we work with.
        """

        raise NotImplementedError

    @typing.override
    @abc.abstractmethod
    def do(self) -> torch.Tensor:
        raise NotImplementedError

    @typing.override
    @abc.abstractmethod
    def deps(self) -> tuple[fn.Fn[object], ...]:
        raise NotImplementedError

    @classmethod
    def from_tensor(cls, data: torch.Tensor, /) -> TensorFn:
        from . import _data

        return _data.TensorDataFn(data)


def tensor(data: TensorFn | torch.Tensor) -> TensorFn:

    if isinstance(data, TensorFn):
        return data

    if isinstance(data, torch.Tensor):
        return TensorFn.from_tensor(data)

    raise TypeError(f"Do not know how to handle {type(data)=}.")


class BasicPreviewFn(TensorFn, abc.ABC):
    def __init__(self):
        super().__init__()

        if not self.deps():
            raise ValueError(
                "Non leaf base only a valid subclass if `deps()` is not empty."
            )

    @typing.override
    @fake.enable_func
    def preview(self) -> attrs.Attr:
        result = self.do()
        assert isinstance(result, torch.Tensor)
        return attrs.attr(result)
