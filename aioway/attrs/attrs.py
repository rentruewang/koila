# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Mapping
from typing import Any, Self

from .devices import Device
from .dtypes import DType
from .shapes import Shape

__all__ = ["Attributes"]


@dcls.dataclass(frozen=True)
class Attributes:
    """
    ``Attributes`` are shared by ``ColumnSchema`` and ``TableSchema``,
    It describes the metadata and what types they are.
    """

    dtype: DType | None = None
    """
    The element data type of a column.
    """

    shape: Shape | None = None
    """
    The shape of the array representation of the column.
    """

    device: Device | None = None
    """
    The device on which the elements of a column are stored.
    """

    def compatible(self, other: Self) -> bool:
        return (
            True
            and _eq_if_not_none(self.dtype, other.dtype)
            and _eq_if_not_none(self.shape, other.shape)
            and _eq_if_not_none(self.device, other.device)
        )

    def to(
        self,
        *,
        dtype: DType | None = None,
        shape: Shape | None = None,
        device: Device | None = None,
    ) -> Self:
        updates: dict[str, Any] = {}

        if dtype is not None:
            updates.update({"dtype": dtype})

        if shape is not None:
            updates.update({"shape": shape})

        if device is not None:
            updates.update({"device": device})

        return dcls.replace(self, **updates)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        dtype = mapping.get("dtype")
        shape = mapping.get("shape")
        device = mapping.get("device")

        return cls(
            dtype=DType.parse(dtype) if dtype is not None else None,
            shape=Shape.from_seq(shape) if shape is not None else None,
            device=Device(device) if device is not None else None,
        )


def _eq_if_not_none[T](left: T | None, right: T | None, /) -> bool:
    return left is None or right is None or left == right
