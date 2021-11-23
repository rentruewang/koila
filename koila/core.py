from __future__ import annotations

import builtins
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    NoReturn,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from numpy import ndarray
from torch import Tensor
from torch.types import Number

from .protocols import CalculateShape, Lazy, LazyFunction, Runnable

T = TypeVar("T")


@dataclass(frozen=True)
class LazyTensor(Lazy[Tensor]):
    def __init__(
        self: TensorLike, data: Tensor | Runnable[Tensor] | Lazy[Tensor]
    ) -> None:
        super().__init__(data)

    # Magic methods

    def __pos__(self) -> TensorLike:
        return LazyTensor.positive(self)

    def __neg__(self) -> TensorLike:
        return LazyTensor.neg(self)

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    def __invert__(self) -> bool:
        return not bool(self)

    def __add__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.add(self, other)

    def __radd__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.add(other, self)

    def __sub__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.sub(self, other)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.sub(other, self)

    def __mul__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.mul(self, other)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.mul(other, self)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.div(self, other)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.div(other, self)

    def __floordiv__(self, other: TensorLike) -> NoReturn:
        will_not_implement(self, other)

    def __rfloordiv__(self, other: TensorLike) -> NoReturn:
        will_not_implement(other, self)

    def __pow__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.pow(self, other)

    def __rpow__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.pow(other, self)

    def __mod__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.fmod(self, other)

    def __rmod__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.fmod(other, self)

    def __divmod__(self, other: TensorLike) -> NoReturn:
        will_not_implement(self, other)

    def __rdivmod__(self, other: TensorLike) -> NoReturn:
        will_not_implement(other, self)

    def __abs__(self) -> TensorLike:
        return LazyTensor.abs(self)

    def __hash__(self) -> int:
        return id(self.data)

    def __matmul__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.matmul(self, other)

    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return LazyTensor.matmul(other, self)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Tuple[TensorLike, ...] = (),
        kwargs: Dict[str, TensorLike] | None = None,
    ) -> TensorLike:
        if kwargs is None:
            kwargs = {}

        if not builtins.all(
            issubclass(typ, (LazyTensor, Tensor, int, float, bool)) for typ in types
        ):
            return NotImplemented

        if (impl := IMPLEMENTED_FUNCTIONS.get(func.__name__, None)) is not None:
            return impl(*args, **kwargs)

        return NotImplemented

    # Arithmetic operations

    @wraps(Tensor.positive)
    def positive(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.positive, self)

    @wraps(Tensor.neg)
    def neg(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.neg, self)

    negative = neg

    @wraps(Tensor.add)
    def add(self: TensorLike, other: TensorLike, *, alpha: float = 1) -> TensorLike:
        return lazy_in_training(torch.add, self, other, alpha=alpha)

    @wraps(Tensor.sub)
    def sub(self: TensorLike, other: TensorLike, *, alpha: float = 1) -> TensorLike:
        return lazy_in_training(torch.sub, self, other, alpha=alpha)

    subtract = sub

    @wraps(Tensor.mul)
    def mul(self: TensorLike, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mul, self, other)

    multiply = mul

    @wraps(Tensor.div)
    def div(self: TensorLike, other: TensorLike, *, rounding_mode=None) -> TensorLike:
        return lazy_in_training(torch.div, self, other, rounding_mode=rounding_mode)

    divide = true_divide = div

    @wraps(Tensor.fmod)
    def fmod(self: TensorLike, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.fmod, self, other)

    remainder = fmod

    @wraps(Tensor.frac)
    def frac(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.frac, self)

    @wraps(Tensor.pow)
    def pow(self: TensorLike, exponent: TensorLike) -> TensorLike:
        return lazy_in_training(torch.pow, self, exponent)

    @wraps(Tensor.exp)
    def exp(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.exp, self)

    @wraps(Tensor.exp2)
    def exp2(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.exp2, self)

    @wraps(Tensor.log)
    def log(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.log, self)

    @wraps(Tensor.log2)
    def log2(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.log2, self)

    @wraps(Tensor.log10)
    def log10(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.log10, self)

    @wraps(Tensor.log1p)
    def log1p(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.log1p, self)

    @wraps(Tensor.abs)
    def abs(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.abs, self)

    absolute = abs

    @wraps(Tensor.matmul)
    def matmul(self: TensorLike, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.matmul, self, other)

    @wraps(Tensor.bmm)
    def bmm(self: TensorLike, mat2: TensorLike) -> TensorLike:
        return lazy_in_training(torch.bmm, self, mat2)

    @wraps(Tensor.mm)
    def mm(self: TensorLike, mat2: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mm, self, mat2)

    @wraps(Tensor.mv)
    def mv(self: TensorLike, vec: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mv, self, vec)

    @wraps(Tensor.dot)
    def dot(self: TensorLike, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.dot, self, other)

    # Statistic operations

    @overload
    def min(self: TensorLike) -> Lazy[Tensor]:
        ...

    @overload
    def min(self: TensorLike, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
        ...

    @overload
    def min(self: TensorLike, other: TensorLike) -> Lazy[Tensor]:
        ...

    @wraps(Tensor.min)
    def min(self: TensorLike, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _min(self, *args, **kwargs)

    @overload
    def max(self: TensorLike) -> Lazy[Tensor]:
        ...

    @overload
    def max(self: TensorLike, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
        ...

    @overload
    def max(self: TensorLike, other: TensorLike) -> Lazy[Tensor]:
        ...

    @wraps(Tensor.max)
    def max(self: TensorLike, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _max(self, *args, **kwargs)

    @overload
    def mean(self: TensorLike) -> TensorLike:
        ...

    @overload
    def mean(
        self: TensorLike, dim: int | Tuple[int, ...], keepdim: bool = False
    ) -> TensorLike:
        ...

    @wraps(Tensor.mean)
    def mean(self: TensorLike, *args: Any, **kwargs: Any) -> TensorLike:
        return lazy_in_training(torch.mean, self, *args, **kwargs)

    @wraps(Tensor.std)
    def std(
        self, dim: int | Tuple[int, ...], unbiased: bool, keepdim: bool = False
    ) -> TensorLike:
        return lazy_in_training(torch.std, self, dim, unbiased, keepdim)

    @wraps(Tensor.minimum)
    def minimum(self: TensorLike, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.minimum, self, other)

    @wraps(Tensor.maximum)
    def maximum(self: TensorLike, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.maximum, self, other)

    @wraps(Tensor.amin)
    def amin(
        self: TensorLike, dim: int | Tuple[int, ...], keepdim: bool = False
    ) -> TensorLike:
        return lazy_in_training(torch.amin, self, dim, keepdim)

    @wraps(Tensor.amax)
    def amax(
        self: TensorLike, dim: int | Tuple[int, ...], keepdim: bool = False
    ) -> TensorLike:
        return lazy_in_training(torch.amax, self, dim, keepdim)

    @wraps(Tensor.argmin)
    def argmin(
        self: TensorLike, dim: int | None = None, keepdim: bool = False
    ) -> TensorLike:
        return lazy_in_training(torch.argmin, self, dim, keepdim)

    @wraps(Tensor.argmax)
    def argmax(
        self: TensorLike, dim: int | None = None, keepdim: bool = False
    ) -> TensorLike:
        return lazy_in_training(torch.argmax, self, dim, keepdim)

    @wraps(Tensor.isclose)
    def isclose(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.isclose, self)

    @wraps(Tensor.allclose)
    def allclose(self: TensorLike, other: TensorLike) -> bool:
        return run(self).allclose(run(other))

    @wraps(Tensor.item)
    def item(self: TensorLike) -> bool | int | float:
        return run(self).item()

    # Shaping functions

    def numel(self: TensorLike) -> int:
        return run(self).numel()

    @overload
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    def size(self, dim: int) -> int:
        ...

    @wraps(Tensor.size)
    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        tensor = run(self)

        if dim is None:
            return tensor.size()
        else:
            return tensor.size(dim)

    @property
    @wraps(size)
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    @wraps(Tensor.dim)
    def dim(self) -> int:
        return run(self).dim()

    @property
    @wraps(dim)
    def ndim(self) -> int:
        return self.dim()

    # Transform functions

    @wraps(Tensor.t)
    def t(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.t, self)

    @property
    @wraps(t)
    def T(self: TensorLike) -> TensorLike:
        return self.t()

    @wraps(Tensor.permute)
    def permute(self: TensorLike, *dims: int) -> TensorLike:
        return lazy_in_training(torch.permute, self, dims)

    @wraps(Tensor.transpose)
    def transpose(self: TensorLike, dim0: int, dim1: int) -> TensorLike:
        return lazy_in_training(torch.transpose, self, dim0, dim1)

    # Trigonometric functions

    @wraps(Tensor.sin)
    def sin(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.sin, self)

    @wraps(Tensor.cos)
    def cos(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.cos, self)

    @wraps(Tensor.tan)
    def tan(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.tan, self)

    @wraps(Tensor.asin)
    def asin(self: TensorLike) -> Tensor:
        return torch.asin(run(self))

    arcsin = asin

    @wraps(Tensor.acos)
    def acos(self: TensorLike) -> Tensor:
        return torch.acos(run(self))

    arccos = acos

    @wraps(Tensor.atan)
    def atan(self: TensorLike) -> Tensor:
        return torch.atan(run(self))

    arctan = atan

    # Hyperbolic functions

    @wraps(Tensor.sinh)
    def sinh(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.sinh, self)

    @wraps(Tensor.cosh)
    def cosh(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.cosh, self)

    @wraps(Tensor.tanh)
    def tanh(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.tanh, self)

    @wraps(Tensor.asinh)
    def asinh(self: TensorLike) -> Tensor:
        return torch.asinh(run(self))

    arcsinh = asinh

    @wraps(Tensor.acosh)
    def acosh(self: TensorLike) -> Tensor:
        return torch.acosh(run(self))

    arccosh = acosh

    @wraps(Tensor.atanh)
    def atanh(self: TensorLike) -> Tensor:
        return torch.atanh(run(self))

    arctanh = atanh

    # Evaluation functions

    @wraps(Tensor.all)
    def all(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.all, self)

    @wraps(Tensor.any)
    def any(self: TensorLike) -> TensorLike:
        return lazy_in_training(torch.any, self)

    def torch(self: LazyTensor) -> Tensor:
        return self.run()

    @wraps(Tensor.numpy)
    def numpy(self: LazyTensor) -> ndarray:
        return self.run().numpy()

    @wraps(Tensor.tolist)
    def tolist(self: LazyTensor) -> List[Any] | Number:
        return self.run().tolist()


TensorLike = Union[Tensor, LazyTensor]


@overload
def lazy(val: Tensor) -> LazyTensor:
    ...


@overload
def lazy(val: LazyTensor) -> LazyTensor:
    ...


@overload
def lazy(val: int) -> Lazy[int]:
    ...


@overload
def lazy(val: float) -> Lazy[float]:
    ...


@overload
def lazy(val: bool) -> Lazy[bool]:
    ...


@overload
def lazy(val: Lazy[int]) -> Lazy[int]:
    ...


@overload
def lazy(val: Lazy[float]) -> Lazy[float]:
    ...


@overload
def lazy(val: Lazy[bool]) -> Lazy[bool]:
    ...


def lazy(val: Any) -> Any:
    if isinstance(val, LazyTensor):
        return val

    if isinstance(val, Tensor):
        return LazyTensor(val)

    return Lazy(val)


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


def lazy_in_training(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> LazyTensor | Tensor:
    if torch.is_grad_enabled():
        return LazyTensor(LazyFunction(func, shape)(*args, **kwargs))  # type: ignore
    else:
        return func(*args, **kwargs)


# Functions that require special handling.


class _ValIdx(NamedTuple):
    values: Tensor
    indices: Tensor


@wraps(torch.min)
def _min(input: TensorLike, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.min)(input, *args, **kwargs)


@overload
def _max(input: TensorLike) -> Lazy[Tensor]:
    ...


@overload
def _max(input: TensorLike, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
    ...


@overload
def _max(input: TensorLike, other: TensorLike) -> Lazy[Tensor]:
    ...


@wraps(torch.max)
def _max(input: TensorLike, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.max)(input, *args, **kwargs)


@wraps(torch.permute)
def _permute(input: TensorLike, dims: int | Tuple[int, ...]) -> TensorLike:
    return lazy_in_training(torch.permute, input, dims)


def will_not_implement(*args: Any, **kwargs: Any) -> NoReturn:
    _ = args
    _ = kwargs
    raise NotImplementedError(
        "Sorry, this function will not be implemented."
        " "
        "Because it causes confusion or is dangerous."
    )


class ShapeFunctions(Enum, CalculateShape):
    identity = lambda: None


IMPLEMENTED_FUNCTIONS: MappingProxyType[str, Callable[..., Any]] = MappingProxyType(
    {
        "positive": LazyTensor.positive,
        "negative": LazyTensor.negative,
        "neg": LazyTensor.neg,
        "add": LazyTensor.add,
        "sub": LazyTensor.sub,
        "subtract": LazyTensor.subtract,
        "mul": LazyTensor.mul,
        "multiply": LazyTensor.multiply,
        "div": LazyTensor.div,
        "divide": LazyTensor.divide,
        "true_divide": LazyTensor.true_divide,
        "fmod": LazyTensor.fmod,
        "remainder": LazyTensor.remainder,
        "frac": LazyTensor.frac,
        "pow": LazyTensor.pow,
        "exp": LazyTensor.exp,
        "exp2": LazyTensor.exp2,
        "log": LazyTensor.log,
        "log2": LazyTensor.log2,
        "log10": LazyTensor.log10,
        "log1p": LazyTensor.log1p,
        "abs": LazyTensor.abs,
        "matmul": LazyTensor.matmul,
        "bmm": LazyTensor.bmm,
        "mm": LazyTensor.mm,
        "mv": LazyTensor.mv,
        "dot": LazyTensor.dot,
        "min": LazyTensor.min,
        "max": LazyTensor.max,
        "mean": LazyTensor.mean,
        "std": LazyTensor.std,
        "minimum": LazyTensor.minimum,
        "maximum": LazyTensor.maximum,
        "amin": LazyTensor.amin,
        "amax": LazyTensor.amax,
        "argmin": LazyTensor.argmin,
        "argmax": LazyTensor.argmax,
        "isclose": LazyTensor.isclose,
        "allclose": LazyTensor.allclose,
        "t": LazyTensor.t,
        "permute": _permute,
        "transpose": LazyTensor.transpose,
        "sin": LazyTensor.sin,
        "cos": LazyTensor.cos,
        "tan": LazyTensor.tan,
        "asin": LazyTensor.asin,
        "acos": LazyTensor.acos,
        "atan": LazyTensor.atan,
        "sinh": LazyTensor.sinh,
        "cosh": LazyTensor.cosh,
        "tanh": LazyTensor.tanh,
        "asinh": LazyTensor.asinh,
        "acosh": LazyTensor.acosh,
        "atanh": LazyTensor.atanh,
        # Functions that will not be implemented.
        "__floordiv__": will_not_implement,
    }
)
