from __future__ import annotations

import builtins
import functools
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from numpy import ndarray
from torch import Size, Tensor
from torch.types import Number

from .protocols import Lazy, LazyFunction, Runnable

T = TypeVar("T")


class LazyTensor(Lazy[Tensor]):
    def __init__(self, data: Tensor | Runnable[Tensor] | Lazy[Tensor]) -> None:
        super().__init__(data)

    # Magic methods

    def __pos__(self) -> TensorLike:
        return lazy_in_training(torch.positive, self)

    def __neg__(self) -> TensorLike:
        return lazy_in_training(torch.neg, self)

    def __add__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.add, self, other)

    def __radd__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.add, other, self)

    def __sub__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.sub, self, other)

    def __rsub__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.sub, other, self)

    def __mul__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mul, self, other)

    def __rmul__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mul, other, self)

    def __truediv__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.div, self, other)

    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.div, other, self)

    def __floordiv__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.div, self, other, rounding_mode="trunc")

    def __rfloordiv__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.div, other, self, rounding_mode="trunc")

    def __pow__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.pow, self, other)

    def __rpow__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.pow, other, self)

    def __mod__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.fmod, self, other)

    def __rmod__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.fmod, other, self)

    def __matmul__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.matmul, self, other)

    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.matmul, other, self)

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
        args: Tuple[TensorLike, ...] = (),
        kwargs: Dict[str, TensorLike] | None = None,
    ) -> TensorLike:
        if kwargs is None:
            kwargs = {}

        if not builtins.all(
            issubclass(typ, (LazyTensor, Tensor, int, float, bool)) for typ in types
        ):
            return NotImplemented

        if func in SPECIAL_HANDLES.keys():
            return SPECIAL_HANDLES[func](*args, **kwargs)

        result = lazy_in_training(func, *args, **kwargs)

        if func in EAGER_FUNCTIONS:
            return lazy(result).run()

        return result

    # Arithmetic operations

    @wraps(Tensor.positive)
    def positive(self) -> TensorLike:
        return lazy_in_training(torch.positive, self)

    @wraps(Tensor.neg)
    def neg(self) -> TensorLike:
        return lazy_in_training(torch.neg, self)

    negative = neg

    @wraps(Tensor.add)
    def add(self, other: TensorLike, *, alpha: float = 1) -> TensorLike:
        return lazy_in_training(torch.add, self, other, alpha=alpha)

    @wraps(Tensor.sub)
    def sub(self, other: TensorLike, *, alpha: float = 1) -> TensorLike:
        return lazy_in_training(torch.sub, self, other, alpha=alpha)

    subtract = sub

    @wraps(Tensor.mul)
    def mul(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mul, self, other)

    multiply = mul

    @wraps(Tensor.div)
    def div(self, other: TensorLike, *, rounding_mode=None) -> TensorLike:
        return lazy_in_training(torch.div, self, other, rounding_mode=rounding_mode)

    divide = true_divide = div

    @wraps(Tensor.remainder)
    def remainder(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.fmod, self, other)

    fmod = remainder

    @wraps(Tensor.frac)
    def frac(self) -> TensorLike:
        return lazy_in_training(torch.frac, self)

    @wraps(Tensor.pow)
    def pow(self, exponent: TensorLike) -> TensorLike:
        return lazy_in_training(torch.pow, self, exponent)

    @wraps(Tensor.exp)
    def exp(self) -> TensorLike:
        return lazy_in_training(torch.exp, self)

    @wraps(Tensor.exp2)
    def exp2(self) -> TensorLike:
        return lazy_in_training(torch.exp2, self)

    @wraps(Tensor.log)
    def log(self) -> TensorLike:
        return lazy_in_training(torch.log, self)

    @wraps(Tensor.log2)
    def log2(self) -> TensorLike:
        return lazy_in_training(torch.log2, self)

    @wraps(Tensor.log10)
    def log10(self) -> TensorLike:
        return lazy_in_training(torch.log10, self)

    @wraps(Tensor.log1p)
    def log1p(self) -> TensorLike:
        return lazy_in_training(torch.log1p, self)

    @wraps(Tensor.abs)
    def abs(self) -> TensorLike:
        return lazy_in_training(torch.abs, self)

    absolute = abs

    @wraps(Tensor.matmul)
    def matmul(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.matmul, self, other)

    @wraps(Tensor.bmm)
    def bmm(self, mat2: TensorLike) -> TensorLike:
        return lazy_in_training(torch.bmm, self, mat2)

    @wraps(Tensor.mm)
    def mm(self, mat2: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mm, self, mat2)

    @wraps(Tensor.mv)
    def mv(self, vec: TensorLike) -> TensorLike:
        return lazy_in_training(torch.mv, self, vec)

    @wraps(Tensor.dot)
    def dot(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.dot, self, other)

    # Statistic operations

    @overload
    def min(self) -> Lazy[Tensor]:
        ...

    @overload
    def min(self, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
        ...

    @overload
    def min(self, other: TensorLike) -> Lazy[Tensor]:
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
    def max(self, other: TensorLike) -> Lazy[Tensor]:
        ...

    @wraps(Tensor.max)
    def max(self, *args: Any, **kwargs: Any) -> Lazy[Any]:
        return _max(self, *args, **kwargs)

    @overload
    def mean(self) -> TensorLike:
        ...

    @overload
    def mean(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> TensorLike:
        ...

    @wraps(Tensor.mean)
    def mean(self, *args: Any, **kwargs: Any) -> TensorLike:
        return lazy_in_training(torch.mean, self, *args, **kwargs)

    @wraps(Tensor.std)
    def std(
        self, dim: int | Tuple[int, ...], unbiased: bool, keepdim: bool = False
    ) -> TensorLike:
        return lazy_in_training(torch.std, self, dim, unbiased, keepdim)

    @wraps(Tensor.minimum)
    def minimum(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.minimum, self, other)

    @wraps(Tensor.maximum)
    def maximum(self, other: TensorLike) -> TensorLike:
        return lazy_in_training(torch.maximum, self, other)

    @wraps(Tensor.amin)
    def amin(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> TensorLike:
        return lazy_in_training(torch.amin, self, dim, keepdim)

    @wraps(Tensor.amax)
    def amax(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> TensorLike:
        return lazy_in_training(torch.amax, self, dim, keepdim)

    @wraps(Tensor.aminmax)
    def aminmax(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> _MinMax:
        return _aminmax(self, dim, keepdim)

    @wraps(Tensor.argmin)
    def argmin(self, dim: int | None = None, keepdim: bool = False) -> TensorLike:
        return lazy_in_training(torch.argmin, self, dim, keepdim)

    @wraps(Tensor.argmax)
    def argmax(self, dim: int | None = None, keepdim: bool = False) -> TensorLike:
        return lazy_in_training(torch.argmax, self, dim, keepdim)

    @wraps(Tensor.item)
    def item(self) -> bool | int | float:
        return self.run().item()

    @wraps(Tensor.isclose)
    def isclose(self) -> TensorLike:
        return lazy_in_training(torch.isclose, self)

    @wraps(Tensor.allclose)
    def allclose(self, other: TensorLike) -> bool:
        return self.run().allclose(run(other))

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
    def t(self) -> TensorLike:
        return lazy_in_training(torch.transpose, self)

    @property
    @wraps(t)
    def T(self) -> TensorLike:
        return self.t()

    @wraps(Tensor.permute)
    def permute(self, *dims: int) -> TensorLike:
        return lazy_in_training(torch.permute, self, dims)

    @wraps(Tensor.transpose)
    def transpose(self, dim0: int, dim1: int) -> TensorLike:
        return lazy_in_training(torch.transpose, self, dim0, dim1)

    # Trigonometric functions

    @wraps(Tensor.sin)
    def sin(self) -> TensorLike:
        return lazy_in_training(torch.sin, self)

    @wraps(Tensor.cos)
    def cos(self) -> TensorLike:
        return lazy_in_training(torch.cos, self)

    @wraps(Tensor.tan)
    def tan(self) -> TensorLike:
        return lazy_in_training(torch.tan, self)

    @wraps(Tensor.asin)
    def asin(self) -> TensorLike:
        return lazy_in_training(torch.asin, self)

    arcsin = asin

    @wraps(Tensor.acos)
    def acos(self) -> TensorLike:
        return lazy_in_training(torch.acos, self)

    arccos = acos

    @wraps(Tensor.atan)
    def atan(self) -> TensorLike:
        return lazy_in_training(torch.atan, self)

    arctan = atan

    # Hyperbolic functions

    @wraps(Tensor.sinh)
    def sinh(self) -> TensorLike:
        return lazy_in_training(torch.sinh, self)

    @wraps(Tensor.cosh)
    def cosh(self) -> TensorLike:
        return lazy_in_training(torch.cosh, self)

    @wraps(Tensor.tanh)
    def tanh(self) -> TensorLike:
        return lazy_in_training(torch.tanh, self)

    @wraps(Tensor.asinh)
    def asinh(self) -> TensorLike:
        return lazy_in_training(torch.asinh, self)

    arcsinh = asinh

    @wraps(Tensor.acosh)
    def acosh(self) -> TensorLike:
        return lazy_in_training(torch.acosh, self)

    arccosh = acosh

    @wraps(Tensor.atanh)
    def atanh(self) -> TensorLike:
        return lazy_in_training(torch.atanh, self)

    arctanh = atanh

    # Evaluation functions

    @wraps(Tensor.all)
    def all(self) -> TensorLike:
        return lazy_in_training(torch.all, self)

    @wraps(Tensor.any)
    def any(self) -> TensorLike:
        return lazy_in_training(torch.any, self)

    def torch(self) -> Tensor:
        return self.run()

    @wraps(Tensor.numpy)
    def numpy(self) -> ndarray:
        return self.run().numpy()

    @wraps(Tensor.tolist)
    def tolist(self) -> List[Any] | Number:
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
        return LazyTensor(LazyFunction(func)(*args, **kwargs))
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


class _MinMax(NamedTuple):
    min: TensorLike
    max: TensorLike


@wraps(torch.aminmax)
def _aminmax(
    input: TensorLike, dim: int | Tuple[int, ...], keepdim: bool = False
) -> _MinMax:
    return _MinMax(lazy(input).amin(dim, keepdim), lazy(input).amax(dim, keepdim))


SPECIAL_HANDLES: Dict[Callable[..., Any], Callable[..., Any]] = {
    torch.min: _min,
    torch.max: _max,
    torch.aminmax: _aminmax,
}

EAGER_FUNCTIONS: Set[Callable[..., Any]] = {torch.allclose}
