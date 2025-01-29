# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import Self

from pandas import DataFrame

from aioway.blocks import TensordictBlock

from .tables import TableVisitor

__all__ = ["LinearTable"]


@dcls.dataclass(frozen=True)
class LinearTable:
    """
    ``SourceTable`` is a ``Table`` that depends on 0 ``Table``s,
    representing an external data source.
    """

    block: TensordictBlock
    """
    The in-memory representation of the current table.
    The block itself is stored in CPU memory.
    """

    batch_size: int
    """
    The batch size to use as the smallest block size.
    """

    device: str = "cpu"
    """
    The device to use when computing.
    """

    def __iter__(self):
        for idx in range(0, len(self.block), self.batch_size):
            yield self.block[idx : idx + self.batch_size].to(self.device)

    def accept[T](self, visitor: TableVisitor[T]) -> T:
        return visitor.linear(self)

    @property
    def sources(self) -> tuple[()]:
        return ()

    def to(self, device: str) -> Self:
        return type(self)(block=self.block, batch_size=self.batch_size, device=device)

    @classmethod
    def from_pandas(
        cls, df: DataFrame, /, batch_size: int, device: str = "cpu"
    ) -> Self:
        return cls(
            TensordictBlock.from_pandas(df), batch_size=batch_size, device=device
        )
