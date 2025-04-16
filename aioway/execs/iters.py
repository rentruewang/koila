# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator
from typing import Any, Self

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError
from aioway.execs import Exec
from aioway.plans import PhysicalPlan

from ._data_loader import DataLoaderAdaptor
from .execs import Exec

if typing.TYPE_CHECKING:
    from aioway.frames import Frame
    from aioway.streams import Stream

__all__ = ["RawIteratorExec", "FrameStreamExec"]


@dcls.dataclass(frozen=True)
class IteratorExec(Exec):
    """
    The base of iterator executors.
    Iterator execs are adaptors converting from ``Iterator`` of ``Block``s into ``Exec``s.
    """

    iterator: Iterator[Block]
    """
    The ``Iterator`` that produces ``Block``s.
    """

    @typing.override
    def __next__(self) -> Block:
        item = next(self.iterator)
        assert isinstance(item, Block)
        return item


@typing.final
@dcls.dataclass(frozen=True)
class RawIteratorExec(IteratorExec, key="RAW_ITER"):
    _attrs: AttrSet = dcls.field(repr=False)
    """
    The schema of the ``Exec``.

    Note:
        This attribute has an underscore prefix
        because dataclasses doens't overwrite abstract properties with attributes.
    """

    @property
    @typing.override
    def attrs(self):
        return self._attrs

    @property
    @typing.override
    def children(self) -> tuple[()]:
        """
        A ``RawIteratorExec`` do not have info about its input type,
        therfore, we cannot keep expanding on the input.
        """

        return ()


@typing.final
@dcls.dataclass(frozen=True)
class FrameStreamExec(IteratorExec, key="FRAME_STREAM"):
    dataset: "Frame | Stream"

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.dataset.attrs

    @classmethod
    def tabular(
        cls,
        dataset: "Frame | Stream",
        opt: DataLoaderAdaptor | dict[str, Any] = DataLoaderAdaptor(),
    ) -> Self:
        opt = DataLoaderAdaptor.parse(opt)
        iterator = opt.iterator_of(dataset)
        return cls(iterator, dataset)

    @property
    @typing.override
    def children(self) -> tuple[PhysicalPlan]:
        return (self.dataset,)


class IteratorExecTypeError(AiowayError, TypeError): ...
