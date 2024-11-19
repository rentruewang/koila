# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing

from aioway.blocks import Buffer
from aioway.schemas import DataType

if typing.TYPE_CHECKING:
    from .dataframes import DataFrame

__all__ = ["Series"]


@dcls.dataclass(frozen=True)
class Context:
    df: "DataFrame"
    name: str


@dcls.dataclass(frozen=True)
class Series:
    ctx: Context

    def execute(self) -> Buffer:
        return self.ctx.df.execute()[self.ctx.name]

    def schema(self) -> DataType:
        return self.ctx.df.schema()[self.ctx.name]
