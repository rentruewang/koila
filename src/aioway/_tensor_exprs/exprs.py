# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from collections import abc as cabc

import tensordict as td
import torch

__all__ = ["TensorDictExpr", "TensorExpr", "TensorExprRhs"]


type TensorExprRhs = TensorExpr | torch.Tensor | int | float | bool


class TensorExpr(abc.ABC):
    __match_args__: typing.ClassVar[tuple[str, ...]]

    def __invert__(self):
        from . import ufuncs

        return ufuncs.TensorExpr1.invert(self)

    def __neg__(self):
        from . import ufuncs

        return ufuncs.TensorExpr1.neg(self)

    def __getitem__(self, key: int | slice | torch.Tensor | TensorExpr):
        from . import gathers

        # typing.Self is symbolic. If key is symbolic, use the 2-ary expression.
        if isinstance(key, TensorExpr):
            return gathers.GatherTensorExpr(self, key)

        # This is a unary expression.
        else:
            return gathers.StaticIndexGatherTensorExpr(self, key)

    def __add__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.add(self, other)

    def __sub__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.sub(self, other)

    def __mul__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.mul(self, other)

    def __truediv__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.truediv(self, other)

    def __floordiv__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.floordiv(self, other)

    def __mod__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.mod(self, other)

    def __pow__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.pow(self, other)

    @typing.no_type_check
    def __eq__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.eq(self, other)

    @typing.no_type_check
    def __ne__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.ne(self, other)

    def __ge__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.ge(self, other)

    def __gt__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.gt(self, other)

    def __le__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.le(self, other)

    def __lt__(self, other: TensorExprRhs):
        from . import ufuncs

        return ufuncs.TensorExpr2.lt(self, other)

    def compute(self):
        return self._compute()

    @abc.abstractmethod
    def _compute(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def _inputs(self) -> tuple[TensorExpr, ...]: ...

    def _return_type(self) -> type[torch.Tensor]:
        return torch.Tensor


class TensorDictExpr(abc.ABC):

    @typing.overload
    def __getitem__(self, key: str, /) -> TensorExpr: ...

    @typing.overload
    def __getitem__(self, key: list[str], /) -> typing.Self: ...

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
    def _compute(self) -> td.TensorDict: ...

    def _return_type(self) -> type[td.TensorDict]:
        return td.TensorDict

    @abc.abstractmethod
    def keys(self) -> cabc.KeysView[str]: ...

    def select(self, *keys: str) -> TensorDictExpr:
        from . import relations

        return relations.SelectTensorDictExpr(self, keys)

    def column(self, key: str) -> TensorExpr:
        from . import data

        return data.ColumnTensorExpr(self, key)

    def zip(self, other: TensorDictExpr | td.TensorDict) -> TensorDictExpr:
        from . import data, relations

        match other:
            case TensorDictExpr():
                return relations.ZipTensorDictExpr(self, other)

            case td.TensorDict():
                return self.zip(other=data.SourceTensorDictExpr(other))

        raise TypeError(f"Does not know how to handle {type(other)=}.")
