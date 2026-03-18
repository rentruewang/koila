# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Self

from torch import device as TorchDevice

from aioway import _logging

from ._terms import Term

__all__ = ["Device", "device", "DeviceLike"]

LOGGER = _logging.get_logger(__name__)


class Device:
    """
    The device that the tensor data resides on (and will be used for compute).
    """

    __match_args__ = ("device",)

    def __init__(self, device: str | TorchDevice) -> None:
        # On top of only needing to store `torch.device`,
        # it also does a check to ensure that if a string is passed,
        # the device is valid, and must follow the "device[:index]" format.
        try:
            self._device = TorchDevice(device)
        except RuntimeError as e:
            raise ValueError("Not a valid `torch.device`.") from e

        LOGGER.debug("Device %s instance created", self)

    @typing.override
    def __hash__(self) -> int:
        return hash(self.device)

    @typing.override
    def __eq__(self, other: object) -> bool:
        match other:
            case TorchDevice():
                return self._device == other
            case Device(device):
                return self._device == device
            case str():
                # Instead of converting `other` with `torch.device`,
                # which may fail, compare the string directly.
                return str(self._device) == other

        return NotImplemented

    @typing.override
    def __repr__(self) -> str:
        return str(self)

    @typing.override
    def __str__(self) -> str:
        return str(self._device)

    @property
    def device(self):
        return self._device

    @property
    def term(self):
        return DeviceTerm.make(self)

    @staticmethod
    def parse(item: DeviceLike) -> Device:
        "Alias to the `device` function so you don't need to import."
        return device(item)


type DeviceLike = str | TorchDevice | Device
"Types convertible to a `Device`."


def device(device: DeviceLike, /) -> Device:
    "The convenient wrapper to create a `Device` from compatible types."

    match device:
        case Device():
            return device
        case str() | TorchDevice():
            return Device(device)
        case _:
            raise TypeError(device)


@dcls.dataclass(frozen=True)
class DeviceTerm(Term[Device]):
    device: Device

    def __invert__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return self

    def __add__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __sub__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __mul__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __truediv__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __floordiv__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __mod__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __pow__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    @typing.no_type_check
    def __eq__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    @typing.no_type_check
    def __ne__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __ge__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __gt__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __le__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def __lt__(self, other: Self | DeviceLike) -> Self:
        return self._matching_device(self, other)

    def unpack(self) -> Device:
        return self.device

    @classmethod
    def make(cls, data: Device) -> Self:
        return cls(data)

    @classmethod
    def parse(cls, item: Self | DeviceLike) -> Device:
        if isinstance(item, DeviceTerm):
            return item.device
        else:
            return Device.parse(item)

    @classmethod
    def _matching_device(cls, l: Self | DeviceLike, r: Self | DeviceLike) -> Self:
        left = cls.parse(l)
        right = cls.parse(r)

        if left != right:
            return NotImplemented

        return cls(left)
