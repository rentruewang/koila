# Copyright (c) AIoWay Authors - All Rights Reserved

"``Dataset`` is base class for all datasets."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import KeysView, Sequence
from typing import ClassVar, NamedTuple, Self

from aioway.attrs import AttrSet
from aioway.attrs.attrs import Attr
from aioway.tables import Table

__all__ = ["Dataset", "DatasetColumnView", "DatasetSelectView", "DatasetViewTypes"]


class Dataset(Table, ABC):
    """
    ``Dataset`` is a shared base class for dataset classes, ``Frame``s and ``Stream``s.
    """

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        "All datasets have the metadta ``attrs`` present."

        ...

    @typing.final
    @typing.override
    def keys(self) -> KeysView[str]:
        return self.attrs.keys()

    @typing.override
    def column(self, key: str) -> "DatasetColumnView[Self]":
        col_type, _ = self.view_types()
        return col_type.from_column(self, key)

    @typing.override
    def select(self, *keys: str) -> "DatasetSelectView[Self]":
        _, select_type = self.view_types()
        return select_type.from_columns(self, *keys)

    @classmethod
    @abc.abstractmethod
    def view_types(cls) -> "DatasetViewTypes":
        """
        The type used to construct ``.column``, ``.select`` views.

        The reason this is not a ``ClassVar`` is purely technical,
        because ``*SelectView``s often inherit from ``Self``,
        making it a circular dependency if it were a ``ClassVar``.
        """

        ...


@dcls.dataclass(frozen=True)
class DatasetView[T: Dataset]:

    dset: T
    "The original dataset that would be used in the view."


@dcls.dataclass(frozen=True)
class DatasetColumnView[T: Dataset = Dataset](DatasetView[T], ABC):
    col: str
    "The column to pick. Must be in the original table."

    def __post_init__(self) -> None:
        _assert_column_in_dataset(self.col, self.dset.attrs)

    @property
    @typing.final
    def attr(self) -> Attr:
        return self.dset.attrs.column(self.col)

    @classmethod
    @abc.abstractmethod
    def from_column(cls, dataset: T, /, column: str) -> Self: ...


@dcls.dataclass(frozen=True)
class DatasetSelectView[T: Dataset = Dataset](Dataset, DatasetView[T], ABC):
    """
    Perform a selection in the table.
    This is a ``View``, which means creation is cheap, but you pay the price in runtime.
    """

    COLUMN_TYPE: ClassVar[type[DatasetColumnView[T]]]
    "The column type associated with the current ``DatasetSelectView``."

    cols: Sequence[str]
    "The columns to select. Should be in the original table."

    def __post_init__(self) -> None:
        for col in self.cols:
            _assert_column_in_dataset(col, self.dset.attrs)

    @property
    @typing.final
    def attrs(self) -> AttrSet:
        return self.dset.attrs.select(*self.cols)

    @typing.final
    def column(self, key: str):
        _assert_column_in_dataset(key, self.attrs)
        return self.COLUMN_TYPE.from_column(self.dset, column=key)

    @typing.final
    def select(self, *keys: str):
        for key in keys:
            _assert_column_in_dataset(key, self.attrs)

        return self.dset.select(*keys)

    @classmethod
    @abc.abstractmethod
    def from_columns(cls, dataset: T, /, *columns: str) -> Self: ...


class DatasetViewTypes[T: Dataset](NamedTuple):
    "The view types."

    column: type[DatasetColumnView[T]]
    "The type used to construct ``.column`` views."

    select: type[DatasetSelectView[T]]
    "The type used to construct ``.select`` views."


def _assert_column_in_dataset(col: str, attrs: AttrSet) -> None:
    if col in attrs.keys():
        return

    raise ValueError(f"Column {col} is not present in the original dataset: {attrs=}")
