# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Sequence
from typing import Self

from aioway.blocks import Block
from aioway.schemas import TableSchema

from .tapes import Tape

if typing.TYPE_CHECKING:
    from .series import Series

__all__ = ["DataFrame"]


@dcls.dataclass(frozen=True)
class DataFrame(ABC):
    _tape: Tape

    def __getitem__(self, idx: str) -> Series:
        from .series import Context, Series

        return Series(Context(self, name=idx))

    @abc.abstractmethod
    def schema(self) -> TableSchema: ...

    @abc.abstractmethod
    def operands(self) -> Sequence[Self]: ...

    @property
    def columns(self) -> list[str]:
        return list(self.schema().keys())

    @abc.abstractmethod
    def execute(self) -> Block: ...
