# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Self

from torch import device as TorchDevice

from aioway._signs import Signature
from aioway._tracking import ModuleApiTracker, logging

from ._terms import Term

__all__ = ["Device", "DeviceLike"]

LOGGER = logging.get_logger(__name__)
TRACKER = ModuleApiTracker(lambda: Device)


type DeviceLike = str | TorchDevice | Device
"Types convertible to a `Device`."


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

    def torch(self):
        """
        Convert to `torch.device`.
        """

        return self._device

    @classmethod
    def parse(cls, device: DeviceLike) -> Device:
        "The convenient wrapper to create a `Device` from compatible types."

        if isinstance(device, cls):
            return device

        if isinstance(device, str | TorchDevice):
            return Device(device)

        raise TypeError(f"Cannot handle {type(device)=}.")


type DeviceTermRhs = DeviceTerm | DeviceLike


@dcls.dataclass(frozen=True)
class DeviceTerm(Term[Device]):
    device: Device

    def __invert__(self) -> Self:
        return self.__identity("__invert__")

    def __neg__(self) -> Self:
        return self.__identity("__neg__")

    def __add__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__add__")

    def __sub__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__sub__")

    def __mul__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__mul__")

    def __truediv__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__truediv__")

    def __floordiv__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__floordiv__")

    def __mod__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__mod__")

    def __pow__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__pow__")

    @typing.no_type_check
    def __eq__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__eq__")

    @typing.no_type_check
    def __ne__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__ne__")

    def __gt__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__gt__")

    def __ge__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__ge__")

    def __lt__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__lt__")

    def __le__(self, other: DeviceTermRhs) -> Self:
        return self.__matching_device(self, other, name="__le__")

    def unpack(self) -> Device:
        return self.device

    @classmethod
    def make(cls, data: Device) -> Self:
        return cls(data)

    @classmethod
    def parse(cls, item: DeviceTermRhs) -> Device:
        if isinstance(item, DeviceTerm):
            return item.device
        else:
            return Device.parse(item)

    def __identity(self, name: str):
        with TRACKER(name=name, signature=Signature(Device, Device)):
            return self

    @classmethod
    def __matching_device(cls, l: DeviceTermRhs, r: DeviceTermRhs, name: str) -> Self:
        with TRACKER(name=name, signature=Signature(Device, Device, Device)):
            return cls.__matching_device_impl(l, r)

    @classmethod
    def __matching_device_impl(cls, l: DeviceTermRhs, r: DeviceTermRhs) -> Self:
        left = cls.parse(l)
        right = cls.parse(r)

        if left != right:
            return NotImplemented

        return cls(left)
