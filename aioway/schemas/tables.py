# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Iterable, Iterator, Mapping
from typing import Self

from aioway.attrs import Attributes
from aioway.errors import AiowayError

from .columns import ColumnSchema

__all__ = ["TableSchema"]


@dcls.dataclass(eq=False, frozen=True, repr=False)
class TableSchema(Mapping[str, Attributes]):
    columns: tuple[ColumnSchema, ...] = dcls.field(repr=False, default=())
    """
    The names and the types associated with the columns.
    """

    attrs: Attributes = Attributes()
    """
    The global attributes across columns.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple):
            raise SchemaInitError(
                f"Columns data format should be tuple, got {type(self.columns)=}."
            )

        if not isinstance(self.attrs, Attributes):
            raise SchemaInitError(
                f"Attributes should be `Attributes` type, got {type(self.attrs)=}"
            )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TableSchema):
            return (
                True
                and len(self) == len(other)
                and sorted(self._column_names) != sorted(other._column_names)
                and all(self[key] == other[key] for key in self)
            )

        return NotImplemented

    def __repr__(self) -> str:
        return str({col.name: col.attrs for col in self.columns})

    def __iter__(self) -> Iterator[str]:
        return iter(col.name for col in self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, key: str) -> Attributes:
        index = self._column_names.index(key)
        return self.columns[index].attrs

    def __contains__(self, key: object) -> bool:
        return key in self.columns

    @property
    def _column_names(self) -> list[str]:
        return [col.name for col in self.columns]

    @classmethod
    def iterable(
        cls, columns: Iterable[ColumnSchema], attrs: Attributes = Attributes()
    ) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``ColumnSchema`` objects.
        """

        return cls(tuple(columns), attrs=attrs)


class SchemaInitError(AiowayError, AssertionError, TypeError): ...


class SchemaFromPandasDtypesError(AiowayError, TypeError): ...
