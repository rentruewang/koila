# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from aioway.blocks import TensorBlock

__all__ = ["BinaryExec", "UnaryExec"]


class UnaryExec(Protocol):
    @abc.abstractmethod
    def __call__(self, block: TensorBlock, /) -> TensorBlock: ...


class BinaryExec(Protocol):
    @abc.abstractmethod
    def __call__(self, left: TensorBlock, right: TensorBlock, /) -> TensorBlock: ...
