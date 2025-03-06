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
    columns: Mapping[str, ColumnSchema] = dcls.field(default_factory=dict)
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

        if self.device and not isinstance(self.device, Device):
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

    def __eq__(self, other: object):
        if isinstance(other, TableSchema):
            return sorted(self.columns) == sorted(other.columns)

        # Do not check devices if RHS is mapping.
        if isinstance(other, Mapping):
            return self.columns == other

        return NotImplemented

    def __or__(self, other: Self) -> Self:
        # Using the logic in ``__and__`` to verify intersection.
        _ = self & other

        return type(self)({**self.columns, **other.columns}, device=self.device)

    def __and__(self, other: Self) -> Self:
        joint = set(self.keys()).intersection(other.keys())

        if not all(self[key] == other[key] for key in joint):
            raise SchemaMergeError(
                f"Schema {self} and {other} has different dtypes on intersecting keys."
            )

        # If global device is specified, and not equal then we cannot merge.
        if self.device and other.device and self.device != other.device:
            raise SchemaMergeError("Device is not equal.")

        return type(self)(
            {key: col for key, col in self.columns.items() if key in joint},
            device=self.device,
        )

    def product(self, other: Self, on: str) -> Self:
        if on not in self:
            raise SchemaKeyError(f"{self.columns=} must contain key={on}")

        if on not in other:
            raise SchemaKeyError(f"{other.columns=} must contain key={on}")

        # Merging here is OK, as ``dict`` update overwrites the left side.
        return self | other

    def project(self, *columns: str) -> Self:
        if not all(col in self for col in columns):
            raise SchemaKeyError(f"Schema {self} does not contain all {columns=}")

        return type(self)({col: self[col] for col in columns}, device=self.device)

    def rename(self, **mapping: str) -> Self:
        if not all(key in self for key in mapping):
            raise SchemaKeyError(f"{self} must be a superset of {list(mapping)}")

        return type(self)(
            {mapping.get(col, col): self[col] for col in {*mapping, *self}}
        )

    def transform(self, target: Self) -> Self:
        return target

    def union(self, other: Self) -> Self:
        if self != other:
            raise SchemaMergeError(f"In union, {self} != {other}.")

        return self

    @classmethod
    def iterable(
        cls, columns: Iterable[NamedColumnSchema], device: Device = Device()
    ) -> Self:
        """
        Creates a ``TableSchema`` object from an iterable of ``ColumnSchema`` objects.
        """

        return cls({name: schema for name, schema in columns}, device=device)


class SchemaInitError(AiowayError, AssertionError, TypeError): ...


class SchemaMergeError(AiowayError, KeyError): ...


class SchemaKeyError(AiowayError, KeyError): ...
