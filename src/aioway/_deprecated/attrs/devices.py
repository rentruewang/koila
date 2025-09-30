# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from typing import Any, Self

from torch import device as TorchDevice

from aioway._errors import AiowayError

__all__ = ["Device"]

LOGGER = logging.getLogger(__name__)


@typing.final
@dcls.dataclass(eq=False, frozen=True)
class Device:
    """
    The device that a ``Block`` would run on.
    """

    name: str = "cpu"
    """
    The name of the device. Defaults to "cpu".
    """

    def __post_init__(self) -> None:
        try:
            TorchDevice(self.name)
        except RuntimeError as re:
            raise DeviceUnparsableError(
                f"Device: {self.name} is not parsable by torch."
            ) from re

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        LOGGER.debug("Computing %s == %s", self, other)

        if isinstance(other, Device):
            return self.name == other.name

        if isinstance(other, str):
            return self == TorchDevice(other)

        if isinstance(other, TorchDevice):
            # Somehting like "cpu" or "cuda"
            if other.index is None:
                device_str = other.type
            # Something like "cuda:1" would have
            # type == "cuda" and index == 1.
            else:
                device_str = f"{other.type}:{other.index}"

            return self.name == device_str

        return NotImplemented

    def to(self, device: str) -> Self:
        return dcls.replace(self, name=device)

    def torch(self) -> TorchDevice:
        return TorchDevice(self.name)

    @typing.overload
    @classmethod
    def parse(cls, device: "str | Device | TorchDevice") -> "Device": ...

    @typing.overload
    @classmethod
    def parse(cls, device: None) -> None: ...

    @classmethod
    def parse(cls, device):
        LOGGER.debug("Parsing %s", device)

        if device is None:
            return None

        if isinstance(device, cls):
            return device

        if isinstance(device, str):
            return cls(device)

        if isinstance(device, TorchDevice):
            return cls(str(device))

        raise DeviceUnparsableError(
            f"Unknown device: {device=}. Must be `str` or `torch.device` or `Device`."
        )


class DeviceUnparsableError(AiowayError, ValueError): ...
