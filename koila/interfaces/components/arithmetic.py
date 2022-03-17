from __future__ import annotations

from abc import abstractmethod
from functools import wraps
from typing import Any, NoReturn, Protocol, Union, runtime_checkable

Numeric = Union[int, float, bool]
"Numeric is a union for `int`, `float`, and `bool`, which are all primitive values in C's sense."


@runtime_checkable
class Arithmetic(Protocol):
    """
    `Arithmetic` is a type that supports arithmetic operations.
    Operations such as +-*/ etc are considered arithmetic, basically everything that can be used on a scalar.

    Inheriting this class, requires half of the methods to be overwritten.
    For example, either overload `add` or `__add__`.
    If `__add__` is overwritten, `add` is implemented automatically using `__add__`, and vice versa.
    The only exception is `eq` and `ne`. They must be manually implemented.
    """

    def __invert__(self) -> Arithmetic:
        "The `not` operator."

        return self.logical_not()

    @abstractmethod
    def logical_not(self) -> Arithmetic:
        "The `not` operator."

        ...

    def __pos__(self) -> Arithmetic:
        "The binary `+` operator."

        return self.pos()

    def pos(self) -> Arithmetic:
        "The binary `+` operator."

        return +self

    def __neg__(self) -> Arithmetic:
        "The unary `-` operator."

        return self.neg()

    def neg(self) -> Arithmetic:
        "The unary `-` operator."

        return -self

    def __add__(self, other: Arithmetic) -> Arithmetic:
        "The `+` operator."

        return Arithmetic.add(self, other)

    def __radd__(self, other: Arithmetic) -> Arithmetic:
        "The `+` operator."

        return Arithmetic.add(other, self)

    def add(self, other: Arithmetic) -> Arithmetic:
        "The `+` operator."

        return self + other

    def __sub__(self, other: Arithmetic) -> Arithmetic:
        "The `-` operator."

        return Arithmetic.sub(self, other)

    def __rsub__(self, other: Arithmetic) -> Arithmetic:
        "The `-` operator."

        return Arithmetic.sub(other, self)

    def sub(self, other: Arithmetic) -> Arithmetic:
        "The `-` operator."

        return self - other

    @wraps(sub)
    def subtract(self, other: Arithmetic) -> Arithmetic:
        return self.sub(other)

    def __mul__(self, other: Arithmetic) -> Arithmetic:
        "The `*` operator."

        return Arithmetic.mul(self, other)

    def __rmul__(self, other: Arithmetic) -> Arithmetic:
        "The `*` operator."

        return Arithmetic.mul(other, self)

    def mul(self, other: Arithmetic) -> Arithmetic:
        "The `*` operator."

        return self * other

    @wraps(mul)
    def multiply(self, other: Arithmetic) -> Arithmetic:
        return self.mul(other)

    def __truediv__(self, other: Arithmetic) -> Arithmetic:
        "The `/` operator."

        return self.div(other)

    def __rtruediv__(self, other: Arithmetic) -> Arithmetic:
        "The `/` operator."

        return other.div(self)

    def __floordiv__(self, other: Arithmetic) -> Arithmetic:
        """
        The `//` operator.
        It should not be implemented because of semantic differences between
        `torch`'s `//` and `numpy`'s `//` operator.
        """

        raise NotImplementedError

    def __rfloordiv__(self, other: Arithmetic) -> Arithmetic:
        """
        The `//` operator.
        It should not be implemented because of semantic differences between
        `torch`'s `//` and `numpy`'s `//` operator.
        """

        raise NotImplementedError

    def div(self, other: Arithmetic) -> Arithmetic:
        "The `/` operator."

        return self / other

    @wraps(div)
    def divide(self, other: Arithmetic) -> Arithmetic:
        return self.div(other)

    @wraps(div)
    def truediv(self, other: Arithmetic) -> Arithmetic:
        return self.div(other)

    def __pow__(self, other: Arithmetic) -> Arithmetic:
        "The `**` operator."

        return self.pow(other)

    def __rpow__(self, other: Arithmetic) -> Arithmetic:
        "The `**` operator."

        return Arithmetic.pow(other, self)

    def pow(self, other: Arithmetic) -> Arithmetic:
        "The `**` operator."

        return self ** other

    def __mod__(self, other: Arithmetic) -> Arithmetic:
        "The `%` operator."

        return self.mod(other)

    def __rmod__(self, other: Arithmetic) -> Arithmetic:
        "The `%` operator."

        return other.mod(self)

    def mod(self, other: Arithmetic) -> Arithmetic:
        "The `%` operator."

        return self % other

    @wraps(mod)
    def fmod(self, other: Arithmetic) -> Arithmetic:
        return self.mod(other)

    @wraps(mod)
    def remainder(self, other: Arithmetic) -> Arithmetic:
        return self.mod(other)

    def __divmod__(self, other: Arithmetic) -> NoReturn:
        "The `divmod` operator is not and should not be implemented."

        raise NotImplementedError

    def __rdivmod__(self, other: Arithmetic) -> NoReturn:
        "The `divmod` operator is not and should not be implemented."

        raise NotImplementedError

    def __abs__(self) -> Arithmetic:
        "The `abs` operator."

        return self.abs()

    def abs(self) -> Arithmetic:
        "The `abs` operator."

        return abs(self)

    def __hash__(self) -> int:
        """
        The `hash` operator.
        Since arithmetic types should be value types, the hashing value depends only on its values.
        """

        return id(self)

    def __matmul__(self, other: Arithmetic) -> Arithmetic:
        "The `@` operator."

        return self.matmul(other)

    def __rmatmul__(self, other: Arithmetic) -> Arithmetic:
        "The `@` operator."

        return other.matmul(self)

    def matmul(self, other: Arithmetic) -> Arithmetic:
        "The `@` operator."

        return self @ other

    def __eq__(self, other: Arithmetic | Numeric | Any) -> Arithmetic | bool:
        "The `==` operator."

        if not isinstance(other, (Arithmetic, int, float, bool)):
            return False

        return self.eq(other)

    @abstractmethod
    def eq(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `==` operator. Variables on both sides of the operator are of the same type."

        return self == other

    def __ne__(self, other: Arithmetic | Numeric | Any) -> Arithmetic | bool:
        "The `!=` operator."

        if not isinstance(other, (Arithmetic, int, float, bool)):
            return True

        return self.ne(other)

    @abstractmethod
    def ne(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `!=` operator. Variables on both sides of the operator are of the same type."

        return self != other

    def __gt__(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `>` operator."

        return self.gt(other)

    def gt(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `>` operator."

        return self > other

    def __ge__(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The >= operator."

        return self.ge(other)

    def ge(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `>=` operator."

        return self >= other

    def __lt__(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `<` operator."

        return self.lt(other)

    def lt(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `<` operator."

        return self < other

    def __le__(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `<=` operator."

        return self.le(other)

    def le(self, other: Arithmetic | Numeric) -> Arithmetic:
        "The `<=` operator."

        return self <= other
