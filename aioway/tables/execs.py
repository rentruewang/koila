# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from aioway.blocks import TensordictBlock

__all__ = ["BinaryExec", "UnaryExec"]


class UnaryExec(Protocol):
    @abc.abstractmethod
    def __call__(self, block: TensordictBlock, /) -> TensordictBlock: ...


class BinaryExec(Protocol):
    @abc.abstractmethod
    def __call__(
        self, left: TensordictBlock, right: TensordictBlock, /
    ) -> TensordictBlock: ...
