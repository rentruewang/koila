from __future__ import annotations

from abc import abstractmethod
from typing import NoReturn, Protocol, Union, runtime_checkable


Numeric = Union[int, float, bool]


@runtime_checkable
class Arithmetic(Protocol):
    """
    Arithmetic is a type that supports arithmetic operations.
    Operations such as +-*/ etc are considered arithmetic, basically everything that can be used on a scalar.

    Inheriting this class, requires half of the methods to be overwritten.
    For example, either overload `add` or `__add__`.
    If `__add__` is overwritten, `add` is implemented automatically using `__add__`, and vice versa.
    The only exception is `eq` and `ne`. They must be manually implemented.
    """

    def __invert__(self) -> Arithmetic:
        return self.logical_not()

    @abstractmethod
    def logical_not(self) -> Arithmetic:
        ...

    def __pos__(self) -> Arithmetic:
        return self.pos()

    def pos(self) -> Arithmetic:
        return +self

    def __neg__(self) -> Arithmetic:
        return self.neg()

    def neg(self) -> Arithmetic:
        return -self

    def __add__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.add(self, other)

    def __radd__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.add(other, self)

    def add(self: Arithmetic, other: Arithmetic) -> Arithmetic:
        return self + other

    def __sub__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.sub(self, other)

    def __rsub__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.sub(other, self)

    def sub(self: Arithmetic, other: Arithmetic) -> Arithmetic:
        return self - other

    subtract = sub

    def __mul__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.mul(self, other)

    def __rmul__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.mul(other, self)

    def mul(self: Arithmetic, other: Arithmetic) -> Arithmetic:
        return self * other

    multiply = mul

    def __truediv__(self, other: Arithmetic) -> Arithmetic:
        return self.div(other)

    def __rtruediv__(self, other: Arithmetic) -> Arithmetic:
        return other.div(self)

    def __floordiv__(self, other: Arithmetic) -> Arithmetic:
        raise NotImplementedError

    def __rfloordiv__(self, other: Arithmetic) -> Arithmetic:
        raise NotImplementedError

    def div(self: Arithmetic, other: Arithmetic) -> Arithmetic:
        return self / other

    divide = truediv = div

    def __pow__(self, other: Arithmetic) -> Arithmetic:
        return self.pow(other)

    def __rpow__(self, other: Arithmetic) -> Arithmetic:
        return Arithmetic.pow(other, self)

    def pow(self: Arithmetic, other: Arithmetic) -> Arithmetic:
        return self ** other

    def __mod__(self, other: Arithmetic) -> Arithmetic:
        return self.mod(other)

    def __rmod__(self, other: Arithmetic) -> Arithmetic:
        return other.mod(self)

    def mod(self, other: Arithmetic) -> Arithmetic:
        return self % other

    fmod = remainder = mod

    def __divmod__(self, other: Arithmetic) -> NoReturn:
        raise NotImplementedError

    def __rdivmod__(self, other: Arithmetic) -> NoReturn:
        raise NotImplementedError

    def __abs__(self) -> Arithmetic:
        return self.abs()

    def abs(self) -> Arithmetic:
        return abs(self)

    def __hash__(self) -> int:
        return id(self)

    def __matmul__(self, other: Arithmetic) -> Arithmetic:
        return self.matmul(other)

    def __rmatmul__(self, other: Arithmetic) -> Arithmetic:
        return other.matmul(self)

    def matmul(self, other: Arithmetic) -> Arithmetic:
        return self @ other

    def __eq__(self, other: Arithmetic | Numeric) -> Arithmetic | bool:
        if not isinstance(other, (Arithmetic, int, float, bool)):
            return False
        return self.eq(other)

    @abstractmethod
    def eq(self, other: Arithmetic | Numeric) -> Arithmetic:
        ...

    def __ne__(self, other: Arithmetic | Numeric) -> Arithmetic | bool:
        if not isinstance(other, (Arithmetic, int, float, bool)):
            return True
        return self.ne(other)

    @abstractmethod
    def ne(self, other: Arithmetic | Numeric) -> Arithmetic:
        ...

    def __gt__(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self.gt(other)

    def gt(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self > other

    def __ge__(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self.ge(other)

    def ge(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self >= other

    def __lt__(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self.lt(other)

    def lt(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self < other

    def __le__(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self.le(other)

    def le(self, other: Arithmetic | Numeric) -> Arithmetic:
        return self <= other
