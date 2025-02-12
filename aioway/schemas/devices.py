# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Any, Protocol, Self

__all__ = ["Device", "HasDevice"]


@dcls.dataclass(eq=False, frozen=True)
class Device:
    """
    The device that a ``Block`` would run on.
    """

    name: str = "cpu"
    """
    The name of the device. Defaults to "cpu".
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.name == other

        if isinstance(other, Device):
            return self.name == other.name

        return NotImplemented

    def to(self, device: str) -> Self:
        return dcls.replace(self, name=device)


@typing.runtime_checkable
class HasDevice(Protocol):
    """
    ``HasDevice`` describes a type with a ``device`` attribute.
    """

    device: Device
