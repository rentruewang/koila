# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Iterable, Iterator, Mapping
from typing import Self

from aioway.errors import AiowayError

from .columns import ColumnSchema, NamedColumnSchema
from .devices import Device

__all__ = ["TableSchema"]


@dcls.dataclass(frozen=True)
class TableSchema(Mapping[str, ColumnSchema]):
    columns: dict[str, ColumnSchema] = dcls.field(default_factory=dict)
    """
    The names and the types associated with the columns.
    """

    device: Device | None = None
    """
    The global device to use, if specified.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.columns, dict):
            raise SchemaInitError(
                f"Columns data format should be tuple, got {type(self.columns)=}."
            )

        if not isinstance(self.device, Device):
            raise SchemaInitError(
                f"The device should be `Device` type, got {type(self.device)=}"
            )

        return NotImplemented

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __getitem__(self, key: str) -> ColumnSchema:
        return self.columns[key]

    def __contains__(self, key: object) -> bool:
        return key in self.columns

    @classmethod
    def iterable(
        cls, columns: Iterable[NamedColumnSchema], device: Device = Device()
    ) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``ColumnSchema`` objects.
        """

        return cls({name: schema for name, schema in columns}, device=device)


class SchemaInitError(AiowayError, AssertionError, TypeError): ...


class SchemaFromPandasDtypesError(AiowayError, TypeError): ...
