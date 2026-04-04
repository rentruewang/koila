# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import functools
import operator
import typing
from collections import abc as cabc

import torch
from torch import _subclasses as tsc

from aioway import _common, fake, fn, meta

__all__ = ["TensorFn", "tensor"]


class TensorFn(fn.Fn[torch.Tensor], abc.ABC):
    """
    `TensorFn` is the `Fn` that would produce a `Tensor`.

    As `Fn` represents lazy computation, the base class acts as a router,
    routing different methods to the corresponding subclasses.

    Subclasses should overwrite these functions:

    1. `do()`: Evaluate and generate the `Tensor`.
    2. `deps()`: The dependent `Fn`, that will be evaluated during `do()`.
    3. `preview()`: The (eager) description of the tensor computation.
    """

    def __len__(self) -> int:
        return self.preview().shape[0]

    def __getitem__(self, key: torch.Tensor | TensorFn) -> TensorFn:
        # Spcial case handling for boolean tensors
        # because `FakeTensor` does not support boolean masking.
        if key.dtype == torch.bool:
            return BooleanTensorThunk(self, key)

        return GatherThunk(self, key)

    def __invert__(self) -> TensorFn:
        return UFunc1Thunk(operator.invert, self)

    def __neg__(self) -> TensorFn:
        return UFunc1Thunk(operator.neg, self)

    def __add__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.add, self, other)

    def __sub__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.sub, self, other)

    def __mul__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.mul, self, other)

    def __truediv__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.truediv, self, other)

    def __floordiv__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.floordiv, self, other)

    def __mod__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.mod, self, other)

    def __pow__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.pow, self, other)

    @typing.no_type_check
    def __eq__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.eq, self, other)

    @typing.no_type_check
    def __ne__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.ne, self, other)

    def __gt__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.gt, self, other)

    def __ge__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.ge, self, other)

    def __lt__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.lt, self, other)

    def __le__(self, other: typing.Any) -> TensorFn:
        return UFunc2Thunk(operator.le, self, other)

    @typing.override
    @typing.final
    def do(self) -> torch.Tensor:
        """
        This is the public function that parent `Fn`s should call,
        when the want to request the values of an `Fn`.

        It handles caching in the normal case, so repeated calling the function means that
        the expensive computation (defined within `forward`) would only be called once.

        When fake mode is enabled, it calls `preview` for a fake tensor,
        which is a preview for the normal computation to save computation cost.

        The reason this is modal with `fake.is_enabled()` as a toggle,
        to make sure `preview` and `forward` can use the same codepath as much as possible,
        in the default case `preview` is `forward` with fake mode on.
        """

        if fake.is_enabled():
            return self.preview()

        else:
            return self.__forward_cache()

    @fake.enable_func
    def preview(self) -> tsc.FakeTensor:
        """
        The `preview` function generates a "preview" for the `Tensor` that would be generated.

        The result type (`FakeTensor`) is used as a worst case analysis of the original `Tensor`.

        In most cases (non leaf operators), this method is just a clone of `forward`,
        which is the default implementation of this function.

        In the following cases it must be modified:

        1. Source tensors, `forward` won't be `FakeTensor`, so conversion is needed.
        2. Operators that cannot be supported by `torch` e.g. boolean  masking.
        """

        result = self.forward()
        assert fake.is_fake_tensor(result)
        return result

    @abc.abstractmethod
    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    @typing.override
    @abc.abstractmethod
    def deps(self) -> tuple[fn.Fn[object], ...]:
        raise NotImplementedError

    @property
    def attr(self):
        return meta.attr(self.__get_fake_or_computed())

    def __get_fake_or_computed(self):
        "If `forward` has already been called, return it. Else return `preview`."

        if self.__forward_cache.is_hit:
            return self.__forward_cache()
        else:
            return self.preview()

    @functools.cached_property
    def __forward_cache(self):
        return _TensorFnCache(self.forward)

    @property
    def shape(self):
        return self.attr.shape

    @property
    def device(self):
        return self.attr.device

    @property
    def dtype(self):
        return self.attr.dtype

    @classmethod
    def from_tensor(cls, data: torch.Tensor, /) -> TensorFn:
        return TensorDataFn(data)


def tensor(data: TensorFn | torch.Tensor) -> TensorFn:

    if isinstance(data, TensorFn):
        return data

    if isinstance(data, torch.Tensor):
        return TensorDataFn(data)

    raise TypeError(f"Do not know how to handle {type(data)=}.")


@typing.final
class _TensorFnCache:
    """
    The cacher for `TensorFn.forward`.

    The reason we use this boilerplate over directly using `functools.cache`,
    `functools.cached_property`, or having a saved `.__result` member for instance,
    is because this is the least assuming.

    `functools.cache` assumes that `self` is hashable.
    `functools.cached_property` cannot inspect whether we have evaluated it or not.
    `.__result` member assumes subclass calls `__init__` properly.

    Since this is saved in a `functools.cached_property`, it can be used on unhashable types,
    yet support inspecting whether we called it or not, and does not need to call `__init__`.
    """

    def __init__(self, func: cabc.Callable[[], torch.Tensor]) -> None:
        self._result: torch.Tensor | None = None
        self._func = func

    def __call__(self) -> torch.Tensor:
        if self._result is None:
            self._result = self._func()

        return self._result

    @property
    def is_hit(self):
        return self._result is not None


@_common.dcls_no_eq
class AnyThunk(TensorFn):
    "Represents some computation that is deferred."

    __match_args__ = "func", "args", "kwargs"

    func: cabc.Callable[..., torch.Tensor]
    args: tuple[fn.Fn[typing.Any], ...]
    kwargs: dict[str, fn.Fn[typing.Any]]

    def __post_init__(self) -> None:
        super().__init__()

        for arg in self.args:
            if not isinstance(arg, fn.Fn):
                raise TypeError(f"{arg} is not a `Thunk`.")

        for key, value in self.kwargs.items():
            if not isinstance(value, fn.Fn):
                raise TypeError(f"{key}={value} is not a `Thunk`")

    def __repr__(self) -> str:
        return _common.format_function(self.func, *self.args, **self.kwargs)

    @typing.override
    def forward(self) -> torch.Tensor:
        args = [arg.do() for arg in self.args]
        kwargs = {key: val.do() for key, val in self.kwargs.items()}
        return self.func(*args, **kwargs)

    @typing.override
    def deps(self):
        return tuple(self._deps())

    def _deps(self) -> cabc.Iterator[fn.Fn[object]]:
        yield from self.args
        yield from self.kwargs.values()


@_common.dcls_no_eq
class UFunc1Thunk(TensorFn):
    """
    Thunk for unary function.
    """

    func: cabc.Callable[[torch.Tensor], torch.Tensor]
    arg: TensorFn

    def __post_init__(self):
        super().__init__()

        if not callable(self.func):
            raise TypeError

        if not isinstance(self.arg, TensorFn):
            raise TypeError

    @typing.override
    def forward(self) -> torch.Tensor:
        return self.func(self.arg.forward())

    @typing.override
    def deps(self):
        # If it's a primitive or `torch.Tensor`, do not recurse.
        return (self.arg,)


type BinaryTensorFnRhs = TensorFn | torch.Tensor | int | float | bool


@_common.dcls_no_eq
class UFunc2Thunk(TensorFn):
    """
    Thunk for binary function.
    """

    func: cabc.Callable[[torch.Tensor, typing.Any], torch.Tensor]
    left: TensorFn
    right: BinaryTensorFnRhs

    def __post_init__(self) -> None:

        if not callable(self.func):
            raise TypeError

        if not isinstance(self.left, TensorFn):
            raise TypeError

        if not isinstance(self.right, TensorFn | torch.Tensor | int | float | bool):
            raise TypeError

        super().__init__()

    @typing.override
    def forward(self) -> torch.Tensor:
        left_do = self.left.forward()

        match right := self.right:
            case TensorFn():
                return self.func(left_do, right.forward())
            case _:
                return self.func(left_do, right)

    @typing.override
    def deps(self):
        return tuple(self._deps())

    def _deps(self):
        # If it's a primitive or `torch.Tensor`, do not recurse.
        yield self.left

        if isinstance(right := self.right, TensorFn):
            yield right


@_common.dcls_no_eq
class GatherThunk(TensorFn):
    tensor: TensorFn | torch.Tensor
    index: TensorFn | torch.Tensor

    def __post_init__(self):
        super().__init__()

        # One of them should be `TensorFn`.
        if not isinstance(self.tensor, TensorFn) and isinstance(self.index, TensorFn):
            raise TypeError

    @typing.override
    def forward(self) -> torch.Tensor:
        tensor = (
            self.tensor.forward() if isinstance(self.tensor, TensorFn) else self.tensor
        )
        index = self.index.forward() if isinstance(self.index, TensorFn) else self.index
        return tensor[index]

    @typing.override
    def deps(self):
        return tuple(self._deps())

    def _deps(self) -> cabc.Iterator[TensorFn]:
        if isinstance(self.tensor, TensorFn):
            yield self.tensor

        if isinstance(self.index, TensorFn):
            yield self.index


@_common.dcls_no_eq
class BooleanTensorThunk(GatherThunk):

    def __post_init__(self):
        if not self.index.dtype != torch.bool:
            raise ValueError

        if self.tensor.shape != self.index.shape:
            raise ValueError(f"{self.tensor.shape=}, {self.index.shape=}.")

    @typing.override
    def preview(self) -> tsc.FakeTensor:
        # In the worse case scenario, the boolean is just all 1s.

        return _as_fake(self.tensor)

    @typing.override
    def forward(self):
        tensor_attr = _maybe_forward(self.tensor)
        index_attr = _maybe_forward(tensor=self.index)
        return tensor_attr[index_attr]


def _as_fake(tensor: torch.Tensor | TensorFn):
    if isinstance(tensor, torch.Tensor):
        return fake.to_fake_tensor(tensor)
    else:
        return tensor.preview()


def _maybe_forward(tensor: torch.Tensor | TensorFn):
    if isinstance(tensor, torch.Tensor):
        return tensor
    else:
        return tensor.forward()


@_common.dcls_no_eq
class TensorDataFn(TensorFn):
    "The `fn.Fn` representing a plain `torch.Tensor`."

    data: torch.Tensor

    @typing.override
    def preview(self):
        return fake.to_fake_tensor(self.data)

    @typing.override
    def forward(self) -> torch.Tensor:
        return self.data

    @typing.override
    def deps(self):
        return ()
