# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Iterator

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs.execs import Exec
from aioway.nodes import NullaryNode

__all__ = ["NullaryExec", "IteratorExec"]


@dcls.dataclass
class NullaryExec(Exec, NullaryNode, ABC):
    """
    ``NullaryExec`` is a base class for all nullary operations.
    """

    @typing.override
    @abc.abstractmethod
    def __next__(self) -> Block: ...

    @property
    @typing.override
    @abc.abstractmethod
    def attrs(self) -> AttrSet: ...


@typing.final
@dcls.dataclass
class IteratorExec(NullaryExec, key="ITER"):
    iterator: Iterator[Block]
    """
    The ``Iterator`` that produces ``Block``s.
    """

    attrs: AttrSet = dcls.field(repr=False)
    """
    The schema of the ``Exec``.

    Note:
        This attribute has an underscore prefix
        because dataclasses doens't overwrite abstract properties with attributes.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.iterator)
        assert isinstance(item, Block)
        item.require_attrs(self.attrs)
        return item


class IteratorExecTypeError(AiowayError, TypeError): ...
