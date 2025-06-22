# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.execs.execs import Exec
from aioway.nodes import UnaryNode

__all__ = ["UnaryExec"]


@dcls.dataclass
class UnaryExec(Exec, UnaryNode, ABC):
    """
    ``UnaryExec`` is a base class for all unary operations.
    """

    child: Exec
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
