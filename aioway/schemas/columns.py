# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

from aioway.attrs import Attributes, Device, DType, Shape

__all__ = ["ColumnSchema"]


@dcls.dataclass(frozen=True)
class ColumnSchema:
    """
    ``ColumnSchema`` refers to the schema a column has.
    """

    name: str
    """
    The name of the column.
    """

    attrs: Attributes
    """
    The attributes of the column.
    """

    @property
    def device(self) -> Device | None:
        return self.attrs.device

    @property
    def shape(self) -> Shape | None:
        return self.attrs.shape

    @property
    def dtype(self) -> DType | None:
        return self.attrs.dtype
