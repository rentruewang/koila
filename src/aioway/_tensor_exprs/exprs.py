# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import KeysView
from typing import ClassVar, Self

from tensordict import TensorDict
from torch import Tensor

__all__ = ["TensorDictExpr", "TensorExpr", "TensorExprRhs"]


type TensorExprRhs = TensorExpr | Tensor | int | float | bool


class TensorExpr(ABC):
    __match_args__: ClassVar[tuple[str, ...]]

    def __invert__(self):
        from .ufuncs import UFuncTensorExpr1

        return UFuncTensorExpr1.invert(self)

    def __neg__(self):
        from .ufuncs import UFuncTensorExpr1

        return UFuncTensorExpr1.neg(self)

    def __getitem__(self, key: int | slice | Tensor | TensorExpr):
        from .gathers import GatherTensorExpr, StaticIndexGatherTensorExpr

        # Self is symbolic. If key is symbolic, use the 2-ary expression.
        if isinstance(key, TensorExpr):
            return GatherTensorExpr(self, key)

        # This is a unary expression.
        else:
            return StaticIndexGatherTensorExpr(self, key)

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

    def compute(self):
        return self._compute()

    @abc.abstractmethod
    def _compute(self) -> Tensor: ...

    @abc.abstractmethod
    def _inputs(self) -> tuple[TensorExpr, ...]: ...

    def _return_type(self) -> type[Tensor]:
        return Tensor


class TensorDictExpr(ABC):

    @typing.overload
    def __getitem__(self, key: str, /) -> TensorExpr: ...

    @typing.overload
    def __getitem__(self, key: list[str], /) -> Self: ...

    def __getitem__(self, key, /):
        match key:
            case str():
                return self.column(key)
            case list() if all(isinstance(i, str) for i in key):
                return self.select(*key)

        raise TypeError(
            "The default implemenetation of `Table.__getitem__` "
            f"does not know how to handle {key=}. "
            "It only supports `key` of type `str` and `list[str]`."
        )

    def compute(self):
        return self._compute()

    @abc.abstractmethod
    def _compute(self) -> TensorDict: ...

    def _return_type(self) -> type[TensorDict]:
        return TensorDict

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    def select(self, *keys: str) -> TensorDictExpr:
        from .relations import SelectTensorDictExpr

        return SelectTensorDictExpr(self, keys)

    def column(self, key: str) -> TensorExpr:
        from .data import ColumnTensorExpr

        return ColumnTensorExpr(self, key)

    def zip(self, other: TensorDictExpr | TensorDict) -> TensorDictExpr:
        from .data import SourceTensorDictExpr
        from .relations import ZipTensorDictExpr

        match other:
            case TensorDictExpr():
                return ZipTensorDictExpr(self, other)

            case TensorDict():
                return self.zip(other=SourceTensorDictExpr(other))

        raise TypeError(f"Does not know how to handle {type(other)=}.")
