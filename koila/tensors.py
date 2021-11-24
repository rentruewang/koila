from __future__ import annotations

import builtins
import dataclasses as dcls
import functools
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
    NoReturn,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    final,
    overload,
)

import torch
from numpy import ndarray
from torch import Tensor

from . import shapes
from .runnables import Runnable, RunnableTensor
from .shapes import ShapeFunction

T = TypeVar("T")
V = TypeVar("V", contravariant=True)


@dataclass(frozen=True)
class LazyFunction(Generic[V]):
    func: Callable[..., Tensor]
    shape_func: ShapeFunction

    def __call__(self, *args: Any, **kwargs: Any) -> LazyTensor:
        lazy_args = tuple(lazy(arg) for arg in args)
        lazy_kwargs = dict((k, lazy(v)) for (k, v) in kwargs.items())

        shape_args = [self.size_or_val(arg) for arg in lazy_args]
        shape_kwargs = {k: self.size_or_val(v) for (k, v) in lazy_kwargs.items()}
        shape = self.shape_func(*shape_args, **shape_kwargs)

        return LazyTensor(Evaluation(self.func, shape, *lazy_args, **lazy_kwargs))

    def __get__(self, obj: V, objtype: Type[V]) -> Callable[..., LazyTensor]:
        assert isinstance(obj, objtype), [type(obj), objtype]
        if obj is None:
            return self
        else:
            return functools.partial(self, obj)

    @staticmethod
    def size_or_val(
        argument: RunnableTensor | Any,
    ) -> Tuple[int, ...] | Any:
        if isinstance(argument, RunnableTensor):
            return argument.size()

        return argument


@dataclass(init=False)
class Evaluation(RunnableTensor):
    func: Callable[..., Tensor]
    shape: Tuple[int, ...]
    args: Tuple[LazyTensor | Tensor | int | float | bool, ...] = dcls.field(
        default_factory=tuple
    )
    kwargs: Dict[str, LazyTensor | Tensor | int | float | bool] = dcls.field(
        default_factory=dict
    )

    def __init__(
        self,
        func: Callable[..., Tensor],
        shape: Tuple[int, ...],
        *args: LazyTensor | Tensor | int | float | bool,
        **kwargs: LazyTensor | Tensor | int | float | bool,
    ):
        self.func = func
        self.shape = shape
        self.args = args
        self.kwargs = kwargs

    def run(self) -> Tensor:
        real_args = [run(arg) for arg in self.args]
        real_kwargs = {k: run(v) for (k, v) in self.kwargs.items()}
        result = self.func(*real_args, **real_kwargs)

        assert result.shape == self.shape

        return result

    def _size_impl(self, dim: int | None = None) -> int | Tuple[int, ...]:
        if dim is not None:
            return self.shape[dim]
        else:
            return self.shape


@final
@dataclass
class LazyTensor(RunnableTensor):
    _data: Tensor | RunnableTensor

    def __init__(self, data: Tensor | LazyTensor | RunnableTensor) -> None:
        if isinstance(data, LazyTensor):
            self._data = data._data
        else:
            self._data = data

    # Implementations

    def run(self) -> Tensor:
        data = self._data
        if isinstance(data, Runnable):
            return data.run()
        return data

    # Magic methods

    def __pos__(self) -> TensorLike:
        return torch.positive(self)  # type: ignore

    def __neg__(self) -> TensorLike:
        return torch.neg(self)  # type: ignore

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    def __invert__(self) -> bool:
        return not bool(self)

    def __add__(self, other: TensorLike) -> TensorLike:
        return torch.add(self, other)  # type: ignore

    def __radd__(self, other: TensorLike) -> TensorLike:
        return torch.add(other, self)  # type: ignore

    def __sub__(self, other: TensorLike) -> TensorLike:
        return torch.sub(self, other)  # type: ignore

    def __rsub__(self, other: TensorLike) -> TensorLike:
        return torch.sub(other, self)  # type: ignore

    def __mul__(self, other: TensorLike) -> TensorLike:
        return torch.mul(self, other)  # type: ignore

    def __rmul__(self, other: TensorLike) -> TensorLike:
        return torch.mul(other, self)  # type: ignore

    def __truediv__(self, other: TensorLike) -> TensorLike:
        return torch.div(self, other)  # type: ignore

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        return torch.div(other, self)  # type: ignore

    def __floordiv__(self, other: TensorLike) -> NoReturn:
        will_not_implement(self, other)  # type: ignore

    def __rfloordiv__(self, other: TensorLike) -> NoReturn:
        will_not_implement(other, self)  # type: ignore

    def __pow__(self, other: TensorLike) -> TensorLike:
        return torch.pow(self, other)  # type: ignore

    def __rpow__(self, other: TensorLike) -> TensorLike:
        return torch.pow(other, self)  # type: ignore

    def __mod__(self, other: TensorLike) -> TensorLike:
        return torch.fmod(self, other)  # type: ignore

    def __rmod__(self, other: TensorLike) -> TensorLike:
        return torch.fmod(other, self)  # type: ignore

    def __divmod__(self, other: TensorLike) -> NoReturn:
        will_not_implement(self, other)

    def __rdivmod__(self, other: TensorLike) -> NoReturn:
        will_not_implement(other, self)

    def __abs__(self) -> TensorLike:
        return torch.abs(self)  # type: ignore

    def __hash__(self) -> int:
        return id(self._data)  # type: ignore

    def __matmul__(self, other: TensorLike) -> TensorLike:
        return torch.matmul(self, other)  # type: ignore

    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return torch.matmul(other, self)  # type: ignore

    def __eq__(self, other: TensorLike) -> TensorLike:
        return torch.eq(self, other)  # type: ignore

    def __ne__(self, other: TensorLike) -> TensorLike:
        return torch.ne(self, other)  # type: ignore

    def __gt__(self, other: TensorLike) -> TensorLike:
        return torch.gt(self, other)  # type: ignore

    def __ge__(self, other: TensorLike) -> TensorLike:
        return torch.ge(self, other)  # type: ignore

    def __lt__(self, other: TensorLike) -> TensorLike:
        return torch.lt(self, other)  # type: ignore

    def __le__(self, other: TensorLike) -> TensorLike:
        return torch.le(self, other)  # type: ignore

    def __getattr__(self, name: str) -> Callable[..., Any]:
        method = getattr(Tensor, name)
        wrapper = functools.wraps(method)

        if (custom_impl := CUSTOM_IMPLS.lookup_method(name)) is not None:
            partial = functools.partial(custom_impl, self)
        elif (shape_impl := SHAPE_IMPLS.lookup_method(name)) is not None:
            partial = functools.partial(lazy_forward, method, shape_impl, self)
        else:
            partial = functools.partial(method, run(self))

        return wrapper(partial)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Sequence[TensorLike] = (),
        kwargs: Dict[str, TensorLike] | None = None,
    ) -> TensorLike:
        if kwargs is None:
            kwargs = {}

        if not builtins.all(
            issubclass(typ, (LazyTensor, Tensor, int, float, bool)) for typ in types
        ):
            return NotImplemented

        name = func.__name__
        if (custom_impl := CUSTOM_IMPLS.lookup_function(name)) is not None:
            return custom_impl(*args, **kwargs)
        elif (shape_impl := SHAPE_IMPLS.lookup_function(name)) is not None:
            return lazy_forward(func, shape_impl, *args, **kwargs)
        else:
            args = [run(arg) for arg in args]
            kwargs = {k: run(v) for (k, v) in kwargs.items()}
            return func(*args, **kwargs)

    def _size_impl(self, dim: int | None = None) -> int | Tuple[int, ...]:
        data = self._data

        if dim is None:
            return data.size()

        return data.size(dim)

    @property
    @wraps(Tensor.size)
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    @property
    @wraps(Tensor.dim)
    def ndim(self) -> int:
        return self.dim()

    @property
    @wraps(Tensor.t)
    def T(self) -> TensorLike:
        return self.t()

    def torch(self) -> Tensor:
        return self.run()

    @wraps(Tensor.numpy)
    def numpy(self) -> ndarray:
        return self.run().numpy()

    @wraps(Tensor.tolist)
    def tolist(self) -> List[Any] | int | float | bool:
        return self.run().tolist()


TensorLike = Union[Tensor, LazyTensor]


@overload
def lazy(val: Tensor) -> LazyTensor:
    ...


@overload
def lazy(val: LazyTensor) -> LazyTensor:
    ...


@overload
def lazy(val: int) -> int:
    ...


@overload
def lazy(val: float) -> float:
    ...


@overload
def lazy(val: bool) -> bool:
    ...


def lazy(val: Any) -> Any:
    if isinstance(val, LazyTensor):
        return val

    if isinstance(val, Tensor):
        return LazyTensor(val)

    return val


@overload
def run(val: LazyTensor) -> Tensor:
    ...


@overload
def run(val: RunnableTensor) -> Tensor:
    ...


@overload
def run(val: Runnable[T]) -> T:
    ...


@overload
def run(val: T) -> T:
    ...


def run(val: Any) -> Any:
    if isinstance(val, Runnable):
        return val.run()

    return val


def lazy_forward(
    func: Callable[..., Any], shape_func: ShapeFunction, *args: Any, **kwargs: Any
) -> LazyTensor | Tensor:
    if torch.is_grad_enabled():
        return LazyTensor(LazyFunction(func, shape_func)(*args, **kwargs))
    else:
        return func(*args, **kwargs)


# Functions that require special handling.


class _ValIdx(NamedTuple):
    values: TensorLike
    indices: TensorLike


@overload
def _min(input: TensorLike) -> TensorLike:
    ...


@overload
def _min(input: TensorLike, dim: int, keepdim: bool = False) -> _ValIdx:
    ...


@overload
def _min(input: TensorLike, other: TensorLike) -> TensorLike:
    ...


@wraps(torch.min)
def _min(input: TensorLike, *args: Any, **kwargs: Any) -> TensorLike | _ValIdx:
    if len(args) == len(kwargs) == 0:
        return lazy_forward(torch.min, shapes.scalar, input)

    if (
        len(args) == 1
        and isinstance((other := args[0]), (Tensor, LazyTensor))
        or len(kwargs) == 1
        and (other := kwargs.get("other", None) is not None)
    ):
        return lazy_forward(torch.minimum, shapes.symmetric, input, other)

    return _ValIdx(
        lazy_forward(torch.amin, shapes.reduce_dims, *args, **kwargs),
        lazy_forward(torch.argmin, shapes.reduce_dims, *args, **kwargs),
    )


@overload
def _max(input: TensorLike) -> TensorLike:
    ...


@overload
def _max(input: TensorLike, dim: int, keepdim: bool = False) -> _ValIdx:
    ...


@overload
def _max(input: TensorLike, other: TensorLike) -> TensorLike:
    ...


@wraps(torch.max)
def _max(input: TensorLike, *args: Any, **kwargs: Any) -> TensorLike | _ValIdx:
    if len(args) == len(kwargs) == 0:
        return lazy_forward(torch.max, shapes.scalar, input)

    if (
        len(args) == 1
        and isinstance((other := args[0]), (Tensor, LazyTensor))
        or len(kwargs) == 1
        and (other := kwargs.get("other", None) is not None)
    ):
        return lazy_forward(torch.maximum, shapes.symmetric, input, other)

    return _ValIdx(
        lazy_forward(torch.amax, shapes.reduce_dims, *args, **kwargs),
        lazy_forward(torch.argmax, shapes.reduce_dims, *args, **kwargs),
    )


def _permute_function_shape(
    input: Tuple[int, ...], dims: int | Tuple[int, ...], *args: Any, **kwargs: Any
) -> Tuple[int, ...]:
    if isinstance(dims, int):
        dims = (dims,)

    return shapes.permute(input, *dims)


def _t_shape(input: Tuple[int, ...], *args: Any, **kwargs: Any) -> Tuple[int, ...]:
    shapes.mute_unused_args(*args, **kwargs)
    return shapes.tranpose(input, 0, 1)


def will_not_implement(*args: Any, **kwargs: Any) -> NoReturn:
    _ = args
    _ = kwargs
    raise NotImplementedError(
        "Sorry, this function will not be implemented."
        " "
        "Because it causes confusion or is dangerous."
    )


@dataclass
class MethodFunction(Generic[T]):
    method: Dict[str, T]
    function: Dict[str, T]

    def lookup_method(self, key: str) -> T | None:
        if key in self.method:
            return self.method[key]
        return self.lookup_function(key)

    def lookup_function(self, key: str) -> T | None:
        return self.function.get(key, None)


CUSTOM_IMPLS = MethodFunction[Callable](
    method={},
    function={
        "min": _min,
        "max": _max,
    },
)

SHAPE_IMPLS = MethodFunction[ShapeFunction](
    method={"permute": shapes.permute},
    function={
        "positive": shapes.identity,
        "negative": shapes.identity,
        "neg": shapes.identity,
        "add": shapes.symmetric,
        "sub": shapes.symmetric,
        "subtract": shapes.symmetric,
        "mul": shapes.symmetric,
        "multiply": shapes.symmetric,
        "div": shapes.symmetric,
        "divide": shapes.symmetric,
        "true_divide": shapes.symmetric,
        "fmod": shapes.symmetric,
        "remainder": shapes.symmetric,
        "frac": shapes.identity,
        "pow": shapes.symmetric,
        "exp": shapes.identity,
        "exp2": shapes.identity,
        "log": shapes.identity,
        "log2": shapes.identity,
        "log10": shapes.identity,
        "log1p": shapes.identity,
        "abs": shapes.identity,
        "matmul": shapes.matmul,
        "bmm": shapes.matmul,
        "mm": shapes.matmul,
        "mv": shapes.matmul,
        "dot": shapes.matmul,
        "eq": shapes.symmetric,
        "equal": shapes.symmetric,
        "ne": shapes.symmetric,
        "not_equal": shapes.symmetric,
        "gt": shapes.symmetric,
        "greater": shapes.symmetric,
        "ge": shapes.symmetric,
        "greater_equal": shapes.symmetric,
        "lt": shapes.symmetric,
        "less": shapes.symmetric,
        "le": shapes.symmetric,
        "less_equal": shapes.symmetric,
        "mean": shapes.reduce_dims,
        "std": shapes.scalar,
        "minimum": shapes.symmetric,
        "maximum": shapes.symmetric,
        "amin": shapes.reduce_dims,
        "amax": shapes.reduce_dims,
        "argmin": shapes.reduce_dims,
        "argmax": shapes.reduce_dims,
        "isclose": shapes.symmetric,
        "allclose": shapes.scalar,
        "t": _t_shape,
        "permute": _permute_function_shape,
        "transpose": shapes.tranpose,
        "sin": shapes.identity,
        "cos": shapes.identity,
        "tan": shapes.identity,
        "asin": shapes.identity,
        "acos": shapes.identity,
        "atan": shapes.identity,
        "sinh": shapes.identity,
        "cosh": shapes.identity,
        "tanh": shapes.identity,
        "asinh": shapes.identity,
        "acosh": shapes.identity,
        "atanh": shapes.identity,
        "sigmoid": shapes.identity,
        "hardsigmoid": shapes.identity,
        "relu": shapes.identity,
        "relu6": shapes.identity,
        "leaky_relu": shapes.identity,
        "binary_cross_entropy": shapes.scalar,
        "binary_cross_entropy_with_logits": shapes.scalar,
        "elu": shapes.identity,
        "gelu": shapes.identity,
        "linear": shapes.linear,
        # Functions that will not be implemented.
        "__floordiv__": will_not_implement,
    },
)
