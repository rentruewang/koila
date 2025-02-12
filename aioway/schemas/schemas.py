# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import itertools
import typing
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, NamedTuple, Self, TypeGuard

from numpy import dtype as DTypeType
from pandas import DataFrame

from aioway.errors import AiowayError

from .shapes import Shape

__all__ = ["Schema"]


class NameDType(NamedTuple):
    """
    ``NameDType`` represents a column in a table, comparable by names.

    This exists to make calling `Sequence.index` work for ``Schema``.
    """

    name: str
    """
    The name of the column.
    """

    dtype: Shape
    """
    The data type of the column.
    """


@typing.final
@dcls.dataclass(eq=False, frozen=True, repr=False)
class Schema(Mapping[str, Shape]):
    _columns: tuple[NameDType, ...] = dcls.field(repr=False)
    """
    The names and the types associated with the columns.
    """

    def __post_init__(self) -> None:
        if len(self.names) != len(self.dtypes):
            raise ValueError(
                "Names and types in schemas must have the same length."
                " "
                f"Got {len(self.names)} and {len(self.dtypes)}"
            )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Schema):
            return (
                True
                and len(self) == len(other)
                and all(s == o for s, o in zip(self.sorted(), other.sorted()))
            )

        # For some other kinds of ``Mapping`` that is not ``Schema``,
        # convert them to ``Schema``.
        if isinstance(other, Mapping):
            other_schema = self.mapping(other)
            return self == other_schema

        return NotImplemented

    def __repr__(self) -> str:
        sb = (f"{name}: {type}" for name, type in self._columns)
        return "{" + ", ".join(sb) + "}"

    def __len__(self) -> int:
        return len(self._columns)

    def __getitem__(self, key: str) -> Shape:
        try:
            return next(col.dtype for col in self._columns if col.name == key)
        except StopIteration:
            raise KeyError(f"Key {key} not found.")

    def __contains__(self, key: object) -> bool:
        return key in self.names

    def __iter__(self) -> Iterator[str]:
        return iter(set(self.names))

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.iterable(itertools.chain(self._columns, other._columns))

    @property
    def names(self) -> list[str]:
        return [col.name for col in self._columns]

    @property
    def dtypes(self) -> list[Shape]:
        return [col.dtype for col in self._columns]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def sorted(self) -> list[NameDType]:
        return sorted(self._columns, key=lambda c: c.name)

    @classmethod
    def iterable(cls, columns: Iterable[NameDType]) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``ColumnSchema`` objects.
        """

        return cls(tuple(columns))

    @classmethod
    def tuples(cls, columns: Iterable[tuple[str, Shape]], /) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``str`` and ``ArrayDType``.
        """

        return cls.iterable((NameDType(name, type) for name, type in columns))

    @classmethod
    def mapping(cls, mapping: Mapping[str, Shape], /) -> Self:
        """
        Creates a ``TableSchema`` object from a ``Mapping[str, ArrayDType]``.


        Args:
            mapping: _description_

        Returns:
            _description_
        """

        return cls.tuples(mapping.items())

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> Self:
        if not _is_list_of_strings(list(df.keys())):
            raise SchemaFromPandasDtypesError(
                "Series `dtypes`'s index should only contain strings."
            )

        if not _is_list_of_dtype(list(df.dtypes)):
            raise SchemaFromPandasDtypesError(
                "Series `dtypes`'s should only contain `np.dtype`s."
            )

        raise NotImplementedError


def _is_list_of_strings(item: Any) -> TypeGuard[list[str]]:
    return isinstance(item, list) and all(isinstance(i, str) for i in item)


def _is_list_of_dtype(item: Any) -> TypeGuard[list[DTypeType]]:
    return isinstance(item, list) and all(isinstance(i, DTypeType) for i in item)


class SchemaFromPandasDtypesError(AiowayError, TypeError): ...
