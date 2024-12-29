# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import itertools
import typing
from collections.abc import Iterable, Iterator, Mapping
from typing import Self

from .columns import ColumnSchema
from .types import DataType

__all__ = ["TableSchema"]


@typing.final
@dcls.dataclass(eq=False, frozen=True)
class TableSchema(Mapping[str, DataType]):
    columns: tuple[ColumnSchema, ...]
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
        if isinstance(other, TableSchema):
            self_ord = self.sorted()
            other_ord = other.sorted()

            return (
                True
                and len(self_ord) == len(other_ord)
                and all(s == o for s, o in zip(self_ord, other_ord))
            )

        if isinstance(other, Mapping):
            other_schema = self.mapping(other)
            return self == other_schema

        return NotImplemented

    def __str__(self) -> str:
        sb = [f"{name}: {type}" for name, type in zip(self.names, self.dtypes)]
        return "{" + ", ".join(sb) + "}"

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, key: str) -> DataType:
        try:
            return next(col.dtype for col in self.columns if col.name == key)
        except StopIteration:
            raise KeyError(f"Key {key} not found.")

    def __contains__(self, key: object) -> bool:
        return key in self.names

    def __iter__(self) -> Iterator[str]:
        return iter(set(self.names))

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.iterable(itertools.chain(self.columns, other.columns))

    @property
    def names(self) -> list[str]:
        return [col.name for col in self.columns]

    @property
    def dtypes(self) -> list[DataType]:
        return [col.dtype for col in self.columns]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def sorted(self) -> list[ColumnSchema]:
        return sorted(self.columns, key=lambda c: c.name)

    @classmethod
    def iterable(cls, columns: Iterable[ColumnSchema]) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``ColumnSchema`` objects.
        """

        return cls(tuple(columns))

    @classmethod
    def tuples(cls, columns: Iterable[tuple[str, DataType]], /) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``str`` and ``DataType``.
        """

        return cls.iterable((ColumnSchema(name, type) for name, type in columns))

    @classmethod
    def mapping(cls, mapping: Mapping[str, DataType], /) -> Self:
        """
        Creates a ``TableSchema`` object from a ``Mapping[str, DataType]``.


        Args:
            mapping: _description_

        Returns:
            _description_
        """

        return cls.tuples(mapping.items())
