# Copyright (c) RenChu Wang - All Rights Reserved

from typing import NamedTuple

from .types import DataType

__all__ = ["NameDtype"]


class NameDtype(NamedTuple):
    """
    ``ColumnType`` represents a column in a table, comparable by names.

    This exists to make calling `Sequence.index` work for ``Schema``.
    """

    name: str
    """
    The name of the column.
    """

    dtype: DataType
    """
    The data type of the column.
    """
