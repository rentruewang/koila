from __future__ import annotations

import builtins
import functools
from functools import wraps
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, overload

import torch
from numpy import ndarray
from numpy.core.fromnumeric import transpose
from torch import Size, Tensor
from torch.types import Number

from .protocols import Lazy, LazyFunction, Runnable


class LazyTensor(Lazy[Tensor]):
    def __init__(self, data: Tensor | Runnable[Tensor] | Lazy[Tensor]) -> None:
        super().__init__(data)

    # Magic methods

    def __str__(self) -> str:
        return f"LazyTensor {{ {self.run()} }}"

    def __pos__(self) -> LazyTensor:
        return _pos(self)

    def __neg__(self) -> LazyTensor:
        return _neg(self)

    def __add__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _add(self, other)

    def __radd__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _add(other, self)

    def __sub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _sub(self, other)

    def __rsub__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _sub(other, self)

    def __mul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mul(self, other)

    def __rmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mul(other, self)

    def __truediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(self, other)

    def __rtruediv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(other, self)

    def __floordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(self, other, rounding_mode="trunc")

    def __rfloordiv__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _div(other, self, rounding_mode="trunc")

    def __pow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _pow(self, other)

    def __rpow__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _pow(other, self)

    def __mod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mod(self, other)

    def __rmod__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mod(other, self)

    def __matmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _matmul(self, other)

    def __rmatmul__(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _matmul(other, self)

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

        return LazyTensor(LazyFunction(func)(*args, **kwargs))

    # Arithmetic operations

    @wraps(Tensor.positive)
    def positive(self) -> LazyTensor:
        return _pos(self)

    @wraps(Tensor.neg)
    def neg(self) -> LazyTensor:
        return _neg(self)

    negative = neg

    @wraps(Tensor.add)
    def add(self, other: Tensor | LazyTensor, *, alpha: float = 1) -> LazyTensor:
        return _add(self, other, alpha=alpha)

    @wraps(Tensor.sub)
    def sub(self, other: Tensor | LazyTensor, *, alpha: float = 1) -> LazyTensor:
        return _sub(self, other, alpha=alpha)

    subtract = sub

    @wraps(Tensor.mul)
    def mul(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mul(self, other)

    multiply = mul

    @wraps(Tensor.div)
    def div(self, other: Tensor | LazyTensor, *, rounding_mode=None) -> LazyTensor:
        return _div(self, other, rounding_mode=rounding_mode)

    divide = true_divide = div

    @wraps(Tensor.remainder)
    def remainder(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _mod(self, other)

    fmod = remainder

    @wraps(Tensor.frac)
    def frac(self) -> LazyTensor:
        return _frac(self)

    @wraps(Tensor.pow)
    def pow(self, exponent: Tensor | LazyTensor) -> LazyTensor:
        return _pow(self, exponent)

    @wraps(Tensor.log)
    def log(self) -> LazyTensor:
        return _log(self)

    @wraps(Tensor.log2)
    def log2(self) -> LazyTensor:
        return _log2(self)

    @wraps(Tensor.log10)
    def log10(self) -> LazyTensor:
        return _log10(self)

    @wraps(Tensor.log1p)
    def log1p(self) -> LazyTensor:
        return _log1p(self)

    @wraps(Tensor.abs)
    def abs(self) -> LazyTensor:
        return _abs(self)

    absolute = abs

    @wraps(Tensor.matmul)
    def matmul(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _matmul(self, other)

    @wraps(Tensor.bmm)
    def bmm(self, mat2: Tensor | LazyTensor) -> LazyTensor:
        return _bmm(self, mat2)

    @wraps(Tensor.mm)
    def mm(self, mat2: Tensor | LazyTensor) -> LazyTensor:
        return _mm(self, mat2)

    @wraps(Tensor.mv)
    def mv(self, vec: Tensor | LazyTensor) -> LazyTensor:
        return _mv(self, vec)

    @wraps(Tensor.dot)
    def dot(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _dot(self, other)

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
        return _mean(self, *args, **kwargs)

    @wraps(Tensor.minimum)
    def minimum(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _minimum(self, other)

    @wraps(Tensor.maximum)
    def maximum(self, other: Tensor | LazyTensor) -> LazyTensor:
        return _maximum(self, other)

    @wraps(Tensor.amin)
    def amin(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> LazyTensor:
        return _amin(self, dim, keepdim)

    @wraps(Tensor.amax)
    def amax(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> LazyTensor:
        return _amax(self, dim, keepdim)

    @wraps(Tensor.aminmax)
    def aminmax(self, dim: int | Tuple[int, ...], keepdim: bool = False) -> _MinMax:
        return _aminmax(self, dim, keepdim)

    @wraps(Tensor.argmin)
    def argmin(self, dim: int | None = None, keepdim: bool = False) -> LazyTensor:
        return _argmin(self, dim, keepdim)

    @wraps(Tensor.argmax)
    def argmax(self, dim: int | None = None, keepdim: bool = False) -> LazyTensor:
        return _argmax(self, dim, keepdim)

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
        return _t(self)

    @property
    @wraps(t)
    def T(self) -> LazyTensor:
        return self.t()

    @wraps(Tensor.permute)
    def permute(self, *dims: int) -> LazyTensor:
        return _permute(self, dims)

    @wraps(Tensor.transpose)
    def transpose(self, dim0: int, dim1: int) -> LazyTensor:
        return _transpose(self, dim0, dim1)

    # Trigonometric functions

    @wraps(Tensor.sin)
    def sin(self) -> LazyTensor:
        return _sin(self)

    @wraps(Tensor.cos)
    def cos(self) -> LazyTensor:
        return _cos(self)

    @wraps(Tensor.tan)
    def tan(self) -> LazyTensor:
        return _tan(self)

    @wraps(Tensor.asin)
    def asin(self) -> LazyTensor:
        return _asin(self)

    arcsin = asin

    @wraps(Tensor.acos)
    def acos(self) -> LazyTensor:
        return _acos(self)

    arccos = acos

    @wraps(Tensor.atan)
    def atan(self) -> LazyTensor:
        return _atan(self)

    arctan = atan

    # Hyperbolic functions

    @wraps(Tensor.sinh)
    def sinh(self) -> LazyTensor:
        return _sinh(self)

    @wraps(Tensor.cosh)
    def cosh(self) -> LazyTensor:
        return _cosh(self)

    @wraps(Tensor.tanh)
    def tanh(self) -> LazyTensor:
        return _tanh(self)

    @wraps(Tensor.asinh)
    def asinh(self) -> LazyTensor:
        return _asinh(self)

    arcsinh = asinh

    @wraps(Tensor.acosh)
    def acosh(self) -> LazyTensor:
        return _acosh(self)

    arccosh = acosh

    @wraps(Tensor.atanh)
    def atanh(self) -> LazyTensor:
        return _atanh(self)

    arctanh = atanh

    # Evaluation functions

    @wraps(Tensor.all)
    def all(self) -> LazyTensor:
        return _all(self)

    @wraps(Tensor.any)
    def any(self) -> LazyTensor:
        return _any(self)

    def torch(self) -> Tensor:
        return self.run()

    @wraps(Tensor.numpy)
    def numpy(self) -> ndarray:
        return self.run().numpy()

    @wraps(Tensor.tolist)
    def tolist(self) -> List[Any] | Number:
        return self.run().tolist()


def _lazy_eval(func: Callable[..., Any], *args: Any, **kwargs: Any) -> LazyTensor:
    return LazyTensor(LazyFunction(func)(*args, **kwargs))


# Arithmetic operations


@wraps(torch.positive)
def _pos(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.positive, input)


positive = _pos


@wraps(torch.neg)
def _neg(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.neg, input)


neg = negative = _neg


@wraps(torch.add)
def _add(
    input: Tensor | LazyTensor, other: Tensor | LazyTensor, *, alpha: float = 1
) -> LazyTensor:
    return _lazy_eval(torch.add, input, other, alpha=alpha)


add = _add


@wraps(torch.sub)
def _sub(
    input: Tensor | LazyTensor, other: Tensor | LazyTensor, *, alpha: float = 1
) -> LazyTensor:
    return _lazy_eval(torch.sub, input, other, alpha=alpha)


sub = subtract = _sub


@wraps(torch.mul)
def _mul(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.mul, input, other)


mul = multiply = _mul


@wraps(torch.div)
def _div(
    input: Tensor | LazyTensor, other: Tensor | LazyTensor, *, rounding_mode=None
) -> LazyTensor:
    return _lazy_eval(torch.div, input, other, rounding_mode=rounding_mode)


div = divide = true_divide = _div


@wraps(torch.fmod)
def _mod(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.fmod, input, other)


fmod = remainder = _mod


@wraps(torch.frac)
def _frac(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.frac, input)


frac = _frac


@wraps(torch.pow)
def _pow(input: Tensor | LazyTensor, exponent: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.pow, input, exponent)


pow = _pow


@wraps(torch.log)
def _log(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.log, input)


log = _log


@wraps(torch.log2)
def _log2(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.log2, input)


log2 = _log2


@wraps(torch.log10)
def _log10(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.log10, input)


log10 = _log10


@wraps(torch.log1p)
def _log1p(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.log1p, input)


log1p = _log1p


@wraps(torch.abs)
def _abs(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.abs, input)


abs = absolute = _abs


@wraps(torch.matmul)
def _matmul(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.matmul, input, other)


matmul = _matmul


@wraps(torch.bmm)
def _bmm(input: Tensor | LazyTensor, mat2: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.bmm, input, mat2)


bmm = _bmm


@wraps(torch.mm)
def _mm(input: Tensor | LazyTensor, mat2: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.mm, input, mat2)


mm = _mm


@wraps(torch.mv)
def _mv(input: Tensor | LazyTensor, vec: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.mv, input, vec)


mv = _mv


@wraps(torch.dot)
def _dot(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.dot, input, other)


dot = _dot


class _ValIdx(NamedTuple):
    values: Tensor
    indices: Tensor


@overload
def _min(input: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


@overload
def _min(input: Tensor | LazyTensor, dim: int, keepdim: bool = False) -> Lazy[_ValIdx]:
    ...


@overload
def _min(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> Lazy[Tensor]:
    ...


@wraps(torch.min)
def _min(input: Tensor | LazyTensor, *args: Any, **kwargs: Any) -> Lazy[Any]:
    return LazyFunction(torch.min)(input, *args, **kwargs)


min = _min


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


max = _max


@overload
def _mean(input: Tensor | LazyTensor) -> LazyTensor:
    ...


@overload
def _mean(
    input: Tensor | LazyTensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> LazyTensor:
    ...


@wraps(torch.mean)
def _mean(input: Tensor | LazyTensor, *args: Any, **kwargs: Any) -> LazyTensor:
    return _lazy_eval(torch.mean, input, *args, **kwargs)


mean = _mean


@wraps(torch.minimum)
def _minimum(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.minimum, input, other)


minimum = _minimum


@wraps(torch.maximum)
def _maximum(input: Tensor | LazyTensor, other: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.maximum, input, other)


maximum = _maximum


@wraps(torch.amin)
def _amin(
    input: Tensor | LazyTensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> LazyTensor:
    return _lazy_eval(torch.amin, input, dim, keepdim)


amin = _amin


@wraps(torch.amax)
def _amax(
    input: Tensor | LazyTensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> LazyTensor:
    return _lazy_eval(torch.amax, input, dim, keepdim)


amax = _amax


class _MinMax(NamedTuple):
    min: LazyTensor
    max: LazyTensor


@wraps(torch.aminmax)
def _aminmax(
    input: Tensor | LazyTensor, dim: int | Tuple[int, ...], keepdim: bool = False
) -> _MinMax:
    return _MinMax(_amin(input, dim, keepdim), _amax(input, dim, keepdim))


aminmax = _aminmax


@wraps(torch.argmin)
def _argmin(
    input: Tensor | LazyTensor, dim: int | None = None, keepdim: bool = False
) -> LazyTensor:
    return _lazy_eval(torch.argmin, input, dim, keepdim)


argmin = _argmin


@wraps(torch.argmax)
def _argmax(
    input: Tensor | LazyTensor, dim: int | None = None, keepdim: bool = False
) -> LazyTensor:
    return _lazy_eval(torch.argmax, input, dim, keepdim)


argmax = _argmax


# Shaping functions


@wraps(torch.t)
def _t(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.t, input)


t = _t


@wraps(torch.permute)
def _permute(input: Tensor | LazyTensor, dims: Tuple[int, ...]) -> LazyTensor:
    return _lazy_eval(torch.permute, input, dims)


permute = _permute


@wraps(torch.transpose)
def _transpose(input: Tensor | LazyTensor, dim0: int, dim1: int) -> LazyTensor:
    return _lazy_eval(torch.transpose, input, dim0, dim1)


transpose = _transpose

# Trigonometric functions


@wraps(torch.sin)
def _sin(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.sin, input)


sin = _sin


@wraps(torch.cos)
def _cos(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.cos, input)


cos = _cos


@wraps(torch.tan)
def _tan(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.tan, input)


tan = _tan


@wraps(torch.asin)
def _asin(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.asin, input)


asin = arcsin = _asin


@wraps(torch.acos)
def _acos(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.acos, input)


acos = arccos = _acos


@wraps(torch.atan)
def _atan(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.atan, input)


atan = arctan = _atan


# Hyperbolic functions


@wraps(torch.sinh)
def _sinh(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.sinh, input)


sinh = _sinh


@wraps(torch.cosh)
def _cosh(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.cosh, input)


cosh = _cosh


@wraps(torch.tanh)
def _tanh(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.tanh, input)


tanh = _tanh


@wraps(torch.asinh)
def _asinh(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.asinh, input)


asinh = arcsinh = _asinh


@wraps(torch.acosh)
def _acosh(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.acosh, input)


acosh = arccosh = _acosh


@wraps(torch.atanh)
def _atanh(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.atanh, input)


atanh = arctanh = _atanh


# Evaluation functions


@wraps(torch.all)
def _all(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.all, input)


all = _all


@wraps(torch.any)
def _any(input: Tensor | LazyTensor) -> LazyTensor:
    return _lazy_eval(torch.any, input)


any = _any
