# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from .blocks import Block


class UnaryExec(Protocol):
    @abc.abstractmethod
    def __call__(self, block: Block, /) -> Block: ...


class BinaryExec(Protocol):
    @abc.abstractmethod
    def __call__(self, left: Block, right: Block, /) -> Block: ...
