from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from typing import NoReturn, Protocol, Tuple, TypeVar, overload

from torch import device as Device
from torch import dtype as DType

Number = TypeVar("Number", int, float)
Numeric = TypeVar("Numeric", int, float, bool)


class TensorLike(Protocol):
    """
    TensorLike is a protocol that mimics PyTorch's Tensor.
    """

    dtype: DType
    device: str | Device

    data: TensorLike

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        return bool(self.item())

    @abstractmethod
    def __int__(self) -> int:
        return int(self.item())

    @abstractmethod
    def __float__(self) -> float:
        return float(self.item())

    @abstractmethod
    def __invert__(self) -> TensorLike:
        return self.logical_not()

    @abstractmethod
    def logical_not(self) -> TensorLike:
        return not self

    @abstractmethod
    def __pos__(self) -> TensorLike:
        return self.pos()

    @abstractmethod
    def pos(self) -> TensorLike:
        return +self

    @abstractmethod
    def __neg__(self) -> TensorLike:
        return self.neg()

    @abstractmethod
    def neg(self) -> TensorLike:
        return -self

    @abstractmethod
    def __add__(self, other: TensorLike) -> TensorLike:
        return TensorLike.add(self, other)

    @abstractmethod
    def __radd__(self, other: TensorLike) -> TensorLike:
        return TensorLike.add(other, self)

    @abstractmethod
    def add(self: TensorLike, other: TensorLike) -> TensorLike:
        return self + other

    @abstractmethod
    def __sub__(self, other: TensorLike) -> TensorLike:
        return TensorLike.sub(self, other)

    @abstractmethod
    def __rsub__(self, other: TensorLike) -> TensorLike:
        return TensorLike.sub(other, self)

    @abstractmethod
    def sub(self: TensorLike, other: TensorLike) -> TensorLike:
        return self - other

    subtract = sub

    @abstractmethod
    def __mul__(self, other: TensorLike) -> TensorLike:
        return TensorLike.mul(self, other)

    @abstractmethod
    def __rmul__(self, other: TensorLike) -> TensorLike:
        return TensorLike.mul(other, self)

    @abstractmethod
    def mul(self: TensorLike, other: TensorLike) -> TensorLike:
        return self * other

    multiply = mul

    @abstractmethod
    def __truediv__(self, other: TensorLike) -> TensorLike:
        return self.div(other)

    @abstractmethod
    def __rtruediv__(self, other: TensorLike) -> TensorLike:
        return other.div(self)

    def __floordiv__(self, other: TensorLike) -> TensorLike:
        raise NotImplementedError

    def __rfloordiv__(self, other: TensorLike) -> TensorLike:
        raise NotImplementedError

    @abstractmethod
    def div(self: TensorLike, other: TensorLike) -> TensorLike:
        return self / other

    divide = truediv = div

    @abstractmethod
    def __pow__(self, other: TensorLike) -> TensorLike:
        return self.pow(other)

    @abstractmethod
    def __rpow__(self, other: TensorLike) -> TensorLike:
        return TensorLike.pow(other, self)

    @abstractmethod
    def pow(self: TensorLike, other: TensorLike) -> TensorLike:
        return self ** other

    @abstractmethod
    def __mod__(self, other: TensorLike) -> TensorLike:
        return self.mod(other)

    @abstractmethod
    def __rmod__(self, other: TensorLike) -> TensorLike:
        return other.mod(self)

    def mod(self, other: TensorLike) -> TensorLike:
        return self % other

    fmod = remainder = mod

    def __divmod__(self, other: TensorLike) -> NoReturn:
        raise NotImplementedError

    def __rdivmod__(self, other: TensorLike) -> NoReturn:
        raise NotImplementedError

    @abstractmethod
    def __abs__(self) -> TensorLike:
        return self.abs()

    @abstractmethod
    def abs(self) -> TensorLike:
        return abs(self)

    def __hash__(self) -> int:
        return id(self)

    @abstractmethod
    def __matmul__(self, other: TensorLike) -> TensorLike:
        return self.matmul(other)

    @abstractmethod
    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return other.matmul(self)

    @abstractmethod
    def matmul(self, other: TensorLike) -> TensorLike:
        return self @ other

    @abstractmethod
    def __eq__(self, other: TensorLike | Numeric) -> TensorLike:
        return self.eq(other)

    @abstractmethod
    def eq(self, other: TensorLike | Numeric) -> TensorLike:
        return self == other

    @abstractmethod
    def __ne__(self, other: TensorLike) -> TensorLike:
        return self.ne(other)

    @abstractmethod
    def ne(self, other: TensorLike | Numeric) -> TensorLike:
        return self != other

    @abstractmethod
    def __gt__(self, other: TensorLike) -> TensorLike:
        return self.gt(other)

    @abstractmethod
    def gt(self, other: TensorLike) -> TensorLike:
        return self > other

    @abstractmethod
    def __ge__(self, other: TensorLike) -> NoReturn:
        return self.ge(other)

    @abstractmethod
    def ge(self, other: TensorLike) -> TensorLike:
        return self >= other

    @abstractmethod
    def __lt__(self, other: TensorLike) -> NoReturn:
        return self.lt(other)

    @abstractmethod
    def lt(self, other: TensorLike) -> TensorLike:
        return self < other

    @abstractmethod
    def __le__(self, other: TensorLike) -> NoReturn:
        return self.le(other)

    @abstractmethod
    def le(self, other: TensorLike) -> TensorLike:
        return self <= other

    @abstractmethod
    def __len__(self) -> int:
        return self.size(0)

    @abstractmethod
    def dim(self) -> int:
        return len(self.size())

    @property
    def ndim(self) -> int:
        return self.dim()

    ndimension = ndim

    @overload
    @abstractmethod
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    @abstractmethod
    def size(self, dim: int) -> int:
        ...

    @abstractmethod
    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    def numel(self) -> int:
        return functools.reduce(operator.mul, self.shape, 1)

    @abstractmethod
    def item(self) -> Numeric:
        ...

    @abstractmethod
    def transpose(self, dim0: int, dim1: int) -> TensorLike:
        ...

    @property
    def T(self) -> TensorLike:
        return self.transpose(0, 1)
