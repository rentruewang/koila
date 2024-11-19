# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import Self, TypeVar

from pandas import DataFrame

from aioway.blocks import Block

from .tables import Table, TableVisitor

__all__ = ["SourceTable"]

T = TypeVar("T")


@dcls.dataclass(frozen=True)
class SourceTable(Table):
    """
    ``SourceTable`` is a ``Table`` that depends on 0 ``Table``s,
    representing an external data source.
    """

    block: Block

    def __call__(self) -> Block:
        return self.block

    def accept(self, visitor: TableVisitor[T]) -> T:
        return visitor.source(self)

    @property
    def sources(self) -> tuple[()]:
        return ()

    @classmethod
    def from_pandas(cls, df: DataFrame, /) -> Self:
        return cls(Block.from_pandas(df))
