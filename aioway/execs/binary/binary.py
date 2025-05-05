# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.execs.execs import Exec

__all__ = ["BinaryExec"]


@dcls.dataclass
class BinaryExec(Exec, ABC):
    left: Exec
    """
    The LHS of the operator.
    """

    right: Exec
    """
    The RHS of the operator.
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
    def children(self) -> tuple[Exec, Exec]:
        return self.left, self.right
