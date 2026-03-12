# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import KeysView
from typing import ClassVar

from tensordict import TensorDict
from torch import Tensor

from aioway._exprs import Expr
from aioway._tables import Table

__all__ = ["TensorDictExpr", "TensorExpr"]

type TensorExprRhs = TensorExpr | Tensor | int | float | bool


class TensorExpr(Expr[Tensor], ABC):
    __match_args__: ClassVar[tuple[str, ...]]

    def __invert__(self):
        from .ufuncs import UFuncTensorExpr1

        return UFuncTensorExpr1.invert(self)

    def __neg__(self):
        from .ufuncs import UFuncTensorExpr1

        return UFuncTensorExpr1.neg(self)

    def __add__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.add(self, other)

    def __sub__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.sub(self, other)

    def __mul__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.mul(self, other)

    def __truediv__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.truediv(self, other)

    def __floordiv__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.floordiv(self, other)

    def __mod__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.mod(self, other)

    def __pow__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.pow(self, other)

    @typing.no_type_check
    def __eq__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.eq(self, other)

    @typing.no_type_check
    def __ne__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.ne(self, other)

    def __ge__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.ge(self, other)

    def __gt__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.gt(self, other)

    def __le__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.le(self, other)

    def __lt__(self, other: TensorExprRhs):
        from .ufuncs import UFuncTensorExpr2

        return UFuncTensorExpr2.lt(self, other)

    @abc.abstractmethod
    def _compute(self) -> Tensor: ...

    @abc.abstractmethod
    def _inputs(self) -> tuple[TensorExpr, ...]: ...

    def _return_type(self) -> type[Tensor]:
        return Tensor


class TensorDictExpr(Expr[TensorDict], Table[TensorExpr], ABC):

    @abc.abstractmethod
    def _compute(self) -> TensorDict: ...

    def _return_type(self) -> type[TensorDict]:
        return TensorDict

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    def select(self, *keys: str) -> TensorDictExpr:
        from .data import SelectTensorDictExpr

        return SelectTensorDictExpr(self, keys)

    def column(self, key: str) -> TensorExpr:
        from .data import GetItemTensorExpr

        return GetItemTensorExpr(self, key)
