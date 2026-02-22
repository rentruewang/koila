# Copyright (c) AIoWay Authors - All Rights Reserved

"``View``s allow selection operations to be done cheaply."

import dataclasses as dcls
import functools
import typing
from collections.abc import Iterator, KeysView, Sequence
from typing import ClassVar

from .tables import Column, Table

__all__ = ["ColumnView", "SelectView"]


@dcls.dataclass(frozen=True)
class View[T: Table]:
    "The view object that uses a ``Table`` as a base."

    table: T
    "The original table that would be used in the view."


@dcls.dataclass(frozen=True)
class ColumnView[T: Table](Column, View[T]):
    """
    Get a single column from the table.

    This is a ``View``, which means creation is cheap, but you pay the price in runtime.
    """

    column: str
    "The column to pick. Must be in the original table."

    def __post_init__(self) -> None:
        # Check that the column is in the table.
        _assert_column_in_table(self.column, self.table)


@dcls.dataclass(frozen=True)
class SelectView[T: Table](Table, View[T]):
    """
    Perform a selection in the table.
    This is a ``View``, which means creation is cheap, but you pay the price in runtime.
    """

    COLUMN_TYPE: ClassVar[type[ColumnView]]

    columns: Sequence[str]
    "The columns to select. Should be in the original table."

    @classmethod
    def __init_subclass__(cls):
        if not issubclass(cls.COLUMN_TYPE, ColumnView):
            raise TypeError(f"{cls.COLUMN_TYPE=} should be a subclass of `ColumnView`.")

    def __post_init__(self) -> None:
        # The columns should be in the original table.
        for col in self.columns:
            _assert_column_in_table(col, self.table)

    @typing.override
    def keys(self) -> KeysView[str]:
        return self._keys_view

    @functools.cached_property
    def _keys_view(self) -> KeysView[str]:
        return _SelectViewKeys(self)

    @typing.override
    def select(self, *keys: str) -> "SelectView[T]":
        for key in keys:
            _assert_column_in_table(key, self)

        return type(self)(table=self.table, columns=keys)

    @typing.override
    def column(self, key: str) -> ColumnView:
        return self.COLUMN_TYPE(self.table, column=key)


def _assert_column_in_table(col: str, table: Table) -> None:
    if col in table.keys():
        return

    raise ValueError(
        f"Column {col} is not present in the original table: {table.keys()}"
    )


@dcls.dataclass(frozen=True)
class _SelectViewKeys(KeysView[str]):
    "The ``KeysView`` object for ``SelectView``."

    select: SelectView
    "The original view object."

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __contains__(self, obj: object) -> bool:
        return obj in self.columns

    @functools.cached_property
    def columns(self) -> frozenset[str]:
        "Convert the columns to a ``frozenset`` for repeated querying."

        return frozenset(self.select.columns)
