# Copyright (c) AIoWay Authors - All Rights Reserved

"The lazy version of ``Chunk``s."

import dataclasses as dcls
import typing
from typing import Self

from numpy import ndarray as NpArr
from torch import Tensor

from aioway import _typing
from aioway._errors import GitHubTicketFiled
from aioway._exprs import Expr
from aioway._tables import Table
from aioway._tensors import (
    BatchTensorDictExpr,
    ItemTensorDictExpr,
    RenameTensorDictExpr,
    TensorDictExpr,
)
from aioway.attrs import AttrSet

from ..vectors import Vector, VectorExpr
from .chunks import Chunk

__all__ = ["ChunkExpr"]


@dcls.dataclass(frozen=True)
class ChunkExpr(Expr[Chunk], Table[VectorExpr]):
    """
    The expression type for ``Chunk``.
    """

    tensordict: TensorDictExpr
    attrs: AttrSet

    def keys(self):
        return self.tensordict.keys()

    @typing.override
    def _compute(self) -> Chunk:
        data = self.tensordict.compute()
        return Chunk(data=data, attrs=self.attrs)

    @typing.overload
    def __getitem__(self, idx: str) -> VectorExpr: ...

    @typing.overload
    def __getitem__(
        self,
        idx: int | slice | list[int] | list[str] | NpArr | Tensor | Vector,
        /,
    ) -> Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.column(idx)

        if isinstance(idx, int):
            return type(self)(
                tensordict=ItemTensorDictExpr(source=self.tensordict, index=idx),
                attrs=self.attrs[idx],
            )

        if isinstance(idx, Vector):
            return self.__getitem__(idx.data)

        # Only supports ``Vector`` now, not ``VectorExpr``.
        if isinstance(idx, VectorExpr):
            raise GitHubTicketFiled(
                209, "ChunkExpr does not yet support expression keys."
            )

        # Doing the batch operations that simply does type check first,
        # to avoid expensive iteration over the input.
        if isinstance(idx, slice | NpArr | Tensor) or _typing.is_list_of(int)(idx):
            return type(self)(
                tensordict=BatchTensorDictExpr(self.tensordict, idx),
                attrs=self.attrs,
            )

        if _typing.is_list_of(str)(idx):
            return self.select(*idx)

        raise TypeError(type(idx))

    @typing.override
    def column(self, key: str) -> VectorExpr:
        tensor = self.tensordict[key]
        attr = self.attrs[key]
        return VectorExpr(tensor=tensor, attr=attr)

    @typing.override
    def select(self, *keys: str) -> Self:
        td = self.tensordict.select(*keys)
        schema = self.attrs.select(*keys)
        return type(self)(tensordict=td, attrs=schema)

    def rename(self, **renames) -> Self:
        td = RenameTensorDictExpr(self.tensordict, renames)
        return type(self)(tensordict=td, attrs=self.attrs.rename(**renames))

    def zip(self, rhs: ChunkExpr | Chunk) -> Self:
        rhs_td = rhs.tensordict if isinstance(rhs, ChunkExpr) else rhs.data
        td = self.tensordict.zip(rhs_td)
        return type(self)(
            tensordict=td, attrs=AttrSet.from_dict({**self.attrs, **rhs.attrs})
        )

    def _inputs(self) -> tuple[Expr, ...]:
        return (self.tensordict,)

    def _return_type(self) -> type[Chunk]:
        return Chunk
