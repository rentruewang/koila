# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.execs.execs import Exec

__all__ = ["UnaryExec"]


@dcls.dataclass
class UnaryExec(Exec, ABC):
    """
    ``UnaryExec`` is a base class for all unary operations.
    """

    exe: Exec
    """
    The input ``Exec`` of the current ``Exec``.
    """

    @typing.override
    @abc.abstractmethod
    def __next__(self) -> Block: ...

    @property
    @typing.override
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...

    @property
    @typing.override
    def children(self) -> tuple[Exec]:
        return (self.exe,)
