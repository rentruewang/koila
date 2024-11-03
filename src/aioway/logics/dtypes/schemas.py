# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Self

from .columns import NamedDataType
from .dtypes import DataType


@dcls.dataclass(frozen=True)
class Schema(Mapping[str, DataType]):
    columns: Sequence[NamedDataType]
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
            self_sorted = self.sorted()
            other_sorted = other.sorted()

            return len(self_sorted) == len(other_sorted) and all(
                s.name == o.name and s.dtype == o.dtype
                for s, o in zip(self_sorted, other_sorted)
            )

        if isinstance(other, Mapping):
            for col in self.columns:
                if other[col.name] != col.dtype:
                    return False
            return True

        return NotImplemented

    def __str__(self) -> str:
        string_builder = [
            f"{name}: {type}" for name, type in zip(self.names, self.dtypes)
        ]
        text = "{" + ", ".join(string_builder) + "}"

        return text

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, key: str) -> DataType:
        for col in self.columns:
            if col.name == key:
                return col.dtype
        raise KeyError(f"Key {key} not found.")

    def __contains__(self, key: object) -> bool:
        return key in self.names

    def __iter__(self) -> Iterator[str]:
        return iter(set(self.names))

    def __or__(self, other: Self) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented

        return type(self).iterable(itertools.chain(self.columns, other.columns))

    @property
    def names(self) -> Sequence[str]:
        return [col.name for col in self.columns]

    @property
    def dtypes(self) -> Sequence[DataType]:
        return [col.dtype for col in self.columns]

    def sorted(self) -> Sequence[NamedDataType]:
        return sorted(self.columns, key=lambda c: c.name)

    @classmethod
    def iterable(cls, columns: Iterable[NamedDataType]) -> Self:
        return cls(tuple(columns))

    @classmethod
    def tuples(cls, columns: Iterable[tuple[str, DataType]], /) -> Self:
        return cls.iterable((NamedDataType(name, type) for name, type in columns))

    @classmethod
    def mapping(cls, mapping: Mapping[str, DataType], /) -> Self:
        return cls.tuples(mapping.items())

    @classmethod
    def null(cls) -> Self:
        return cls([])
