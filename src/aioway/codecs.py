# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
from abc import ABC
from typing import Protocol

from torch import Tensor

__all__ = ["Encoder", "Decoder", "Codec"]


class Typed[T](Protocol):
    @property
    def data_type(self) -> type[T]: ...


class Encoder[T](Typed[T], ABC):

    @abc.abstractmethod
    def encode(self, data: T) -> Tensor: ...


class Decoder[T](Typed[T], ABC):

    @abc.abstractmethod
    def decode(self, data: Tensor) -> T: ...


class Codec[T](Encoder[T], Decoder[T], Typed[T], ABC): ...
