from __future__ import annotations

import builtins
import functools
from functools import wraps
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, TypeVar, overload

import torch
from numpy import ndarray
from torch import Size, Tensor
from torch.types import Number

from .protocols import Lazy, LazyFunction, Runnable

T = TypeVar("T")


def lazy_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> LazyTensor:
    return LazyTensor(LazyFunction(func)(*args, **kwargs))


# Functions that require special handling.


class _ValIdx(NamedTuple):
    values: Tensor
    indices: Tensor


@wraps(torch.min)
def _min(input: Tensor | LazyTensor, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.min)(input, *args, **kwargs)


@overload
def _max(input: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


@overload
def _max(input: Tensor | LazyTensor, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
    ...


@overload
def _max(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


@wraps(torch.max)
def _max(input: Tensor | LazyTensor, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.max)(input, *args, **kwargs)


class _MinMax(NamedTuple):
    min: LazyTensor
    max: LazyTensor


@wraps(torch.aminmax)
def _aminmax(
    input: Tensor | LazyTensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> _MinMax:
    return _MinMax(lazy(input).amin(dim, keepdim), lazy(input).amax(dim, keepdim))


SPECIAL_HANDLES: Dict[Callable[..., Any], Callable[..., Any]] = {
    torch.min: _min,
    torch.max: _max,
    torch.aminmax: _aminmax,
}


class LazyTensor(Lazy[Tensor]):
    def __init__(self, data: Tensor | Runnable[Tensor] | Lazy[Tensor]) -> None:
        super().__init__(data)

    # Magic methods

    def __str__(self) -> str:
        return f"LazyTensor {{ {self.run()} }}"

    def __pos__(self) -> LazyTensor:
        return lazy_call(torch.positive, self)

    def __neg__(self) -> LazyTensor:
        return lazy_call(torch.neg, self)

    def __add__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.add, self, other)

    def __radd__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.add, other, self)

    def __sub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.sub, self, other)

    def __rsub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.sub, other, self)

    def __mul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.mul, self, other)

    def __rmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.mul, other, self)

    def __truediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.div, self, other)

    def __rtruediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.div, other, self)

    def __floordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.div, self, other, rounding_mode="trunc")

    def __rfloordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.div, other, self, rounding_mode="trunc")

    def __pow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.pow, self, other)

    def __rpow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.pow, other, self)

    def __mod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.fmod, self, other)

    def __rmod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.fmod, other, self)

    def __matmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.matmul, self, other)

    def __rmatmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.matmul, other, self)

    def __getattr__(self, name: str) -> LazyFunction:
        func = getattr(torch, name)
        method = getattr(Tensor, name)
        wrapper = functools.wraps(method)
        partial = functools.partial(func, self)
        return LazyFunction(wrapper(partial))

    def __bool__(self) -> bool:
        return bool(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., Tensor],
        types: Tuple[Type[Any], ...],
        args: Tuple[Tensor | LazyTensor, ...] = (),
        kwargs: Dict[str, Tensor | LazyTensor] | None = None,
    ) -> LazyTensor:
        if kwargs is None:
            kwargs = {}

        if not builtins.all(
            issubclass(typ, (LazyTensor, Tensor, int, float, bool)) for typ in types
        ):
            return NotImplemented

        if func in SPECIAL_HANDLES.keys():
            return SPECIAL_HANDLES[func](*args, **kwargs)

        return lazy_call(func, *args, **kwargs)

    # Arithmetic operations

    @wraps(Tensor.positive)
    def positive(self) -> LazyTensor:
        return lazy_call(torch.positive, self)

    @wraps(Tensor.neg)
    def neg(self) -> LazyTensor:
        return lazy_call(torch.neg, self)

    negative = neg

    @wraps(Tensor.add)
    def add(self, other: Tensor | LazyTensor, *, alpha: float = 1) -> LazyTensor:
        return lazy_call(torch.add, self, other, alpha=alpha)

    @wraps(Tensor.sub)
    def sub(self, other: Tensor | LazyTensor, *, alpha: float = 1) -> LazyTensor:
        return lazy_call(torch.sub, self, other, alpha=alpha)

    subtract = sub

    @wraps(Tensor.mul)
    def mul(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.mul, self, other)

    multiply = mul

    @wraps(Tensor.div)
    def div(self, other: Tensor | LazyTensor, *, rounding_mode=None) -> LazyTensor:
        return lazy_call(torch.div, self, other, rounding_mode=rounding_mode)

    divide = true_divide = div

    @wraps(Tensor.remainder)
    def remainder(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.fmod, self, other)

    fmod = remainder

    @wraps(Tensor.frac)
    def frac(self) -> LazyTensor:
        return lazy_call(torch.frac, self)

    @wraps(Tensor.pow)
    def pow(self, exponent: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.pow, self, exponent)

    @wraps(Tensor.log)
    def log(self) -> LazyTensor:
        return lazy_call(torch.log, self)

    @wraps(Tensor.log2)
    def log2(self) -> LazyTensor:
        return lazy_call(torch.log2, self)

    @wraps(Tensor.log10)
    def log10(self) -> LazyTensor:
        return lazy_call(torch.log10, self)

    @wraps(Tensor.log1p)
    def log1p(self) -> LazyTensor:
        return lazy_call(torch.log1p, self)

    @wraps(Tensor.abs)
    def abs(self) -> LazyTensor:
        return lazy_call(torch.abs, self)

    absolute = abs

    @wraps(Tensor.matmul)
    def matmul(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.matmul, self, other)

    @wraps(Tensor.bmm)
    def bmm(self, mat2: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.bmm, self, mat2)

    @wraps(Tensor.mm)
    def mm(self, mat2: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.mm, self, mat2)

    @wraps(Tensor.mv)
    def mv(self, vec: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.mv, self, vec)

    @wraps(Tensor.dot)
    def dot(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.dot, self, other)

    # Slicing operations

    @overload
    def min(self) -> Lazy[Tensor]:
        ...

    @overload
    def min(self, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
        ...

    @overload
    def min(self, other: Tensor | LazyTensor) -> Lazy[Tensor]:
        ...

    @wraps(Tensor.min)
    def min(self, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _min(self, *args, **kwargs)

    @overload
    def max(self) -> Lazy[Tensor]:
        ...

    @overload
    def max(self, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
        ...

    @overload
    def max(self, other: Tensor | LazyTensor) -> Lazy[Tensor]:
        ...

    @wraps(Tensor.max)
    def max(self, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _max(self, *args, **kwargs)

    @overload
    def mean(self) -> LazyTensor:
        ...

    @overload
    def mean(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> LazyTensor:
        ...

    @wraps(torch.mean)
    def mean(self, *args: Any, **kwargs: Any) -> LazyTensor:
        return lazy_call(torch.mean, self, *args, **kwargs)

    @wraps(Tensor.minimum)
    def minimum(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.minimum, self, other)

    @wraps(Tensor.maximum)
    def maximum(self, other: Tensor | LazyTensor) -> LazyTensor:
        return lazy_call(torch.maximum, self, other)

    @wraps(Tensor.amin)
    def amin(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> LazyTensor:
        return lazy_call(torch.amin, self, dim, keepdim)

    @wraps(Tensor.amax)
    def amax(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> LazyTensor:
        return lazy_call(torch.amax, self, dim, keepdim)

    @wraps(Tensor.aminmax)
    def aminmax(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> _MinMax:
        return _aminmax(self, dim, keepdim)

    @wraps(Tensor.argmin)
    def argmin(self, dim: int | None = None, keepdim: bool = False) -> LazyTensor:
        return lazy_call(torch.argmin, self, dim, keepdim)

    @wraps(Tensor.argmax)
    def argmax(self, dim: int | None = None, keepdim: bool = False) -> LazyTensor:
        return lazy_call(torch.argmax, self, dim, keepdim)

    @wraps(Tensor.item)
    def item(self) -> bool | int | float:
        return self.run().item()

    # Shaping functions

    @overload
    def size(self) -> Size:
        ...

    @overload
    def size(self, dim: int) -> int:
        ...

    @wraps(Tensor.size)
    def size(self, dim: int | None = None) -> int | Size:
        tensor = self.run()
        if dim is None:
            return tensor.dim()
        else:
            return tensor.size(dim)

    @property
    @wraps(size)
    def shape(self) -> Size:
        return self.size()

    @wraps(Tensor.dim)
    def dim(self) -> int:
        return self.run().dim()

    @property
    @wraps(dim)
    def ndim(self) -> int:
        return self.dim()

    @wraps(Tensor.t)
    def t(self) -> LazyTensor:
        return lazy_call(torch.transpose, self)

    @property
    @wraps(t)
    def T(self) -> LazyTensor:
        return self.t()

    @wraps(Tensor.permute)
    def permute(self, *dims: int) -> LazyTensor:
        return lazy_call(torch.permute, self, dims)

    @wraps(Tensor.transpose)
    def transpose(self, dim0: int, dim1: int) -> LazyTensor:
        return lazy_call(torch.transpose, self, dim0, dim1)

    # Trigonometric functions

    @wraps(Tensor.sin)
    def sin(self) -> LazyTensor:
        return lazy_call(torch.sin, self)

    @wraps(Tensor.cos)
    def cos(self) -> LazyTensor:
        return lazy_call(torch.cos, self)

    @wraps(Tensor.tan)
    def tan(self) -> LazyTensor:
        return lazy_call(torch.tan, self)

    @wraps(Tensor.asin)
    def asin(self) -> LazyTensor:
        return lazy_call(torch.asin, self)

    arcsin = asin

    @wraps(Tensor.acos)
    def acos(self) -> LazyTensor:
        return lazy_call(torch.acos, self)

    arccos = acos

    @wraps(Tensor.atan)
    def atan(self) -> LazyTensor:
        return lazy_call(torch.atan, self)

    arctan = atan

    # Hyperbolic functions

    @wraps(Tensor.sinh)
    def sinh(self) -> LazyTensor:
        return lazy_call(torch.sinh, self)

    @wraps(Tensor.cosh)
    def cosh(self) -> LazyTensor:
        return lazy_call(torch.cosh, self)

    @wraps(Tensor.tanh)
    def tanh(self) -> LazyTensor:
        return lazy_call(torch.tanh, self)

    @wraps(Tensor.asinh)
    def asinh(self) -> LazyTensor:
        return lazy_call(torch.asinh, self)

    arcsinh = asinh

    @wraps(Tensor.acosh)
    def acosh(self) -> LazyTensor:
        return lazy_call(torch.acosh, self)

    arccosh = acosh

    @wraps(Tensor.atanh)
    def atanh(self) -> LazyTensor:
        return lazy_call(torch.atanh, self)

    arctanh = atanh

    # Evaluation functions

    @wraps(Tensor.all)
    def all(self) -> LazyTensor:
        return lazy_call(torch.all, self)

    @wraps(Tensor.any)
    def any(self) -> LazyTensor:
        return lazy_call(torch.any, self)

    def torch(self) -> Tensor:
        return self.run()

    @wraps(Tensor.numpy)
    def numpy(self) -> ndarray:
        return self.run().numpy()

    @wraps(Tensor.tolist)
    def tolist(self) -> List[Any] | Number:
        return self.run().tolist()


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
