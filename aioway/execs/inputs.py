# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator
from typing import Any, Self

from aioway.blocks import Block
from aioway.datatypes import AttrSet
from aioway.errors import AiowayError
from aioway.execs import Exec

from ._data_loader import DataLoaderAdaptor
from .execs import Exec

if typing.TYPE_CHECKING:
    from aioway.frames import Frame
    from aioway.streams import Stream

__all__ = ["IteratorExec"]


@dcls.dataclass(frozen=True)
class IteratorExec(Exec, key="ITER"):
    """
    ``IteratorExec`` is an adaptor that converts
    from an ``Iterator`` of ``Block``s into an ``Exec``.
    """

    iterator: Iterator[Block]
    """
    The ``Iterator`` that produces ``Block``s.
    """

    _attrs: AttrSet = dcls.field(repr=False)
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
        return item

    @property
    @typing.override
    def attrs(self):
        return self._attrs

    @classmethod
    def tabular(
        cls,
        dataset: "Frame | Stream",
        opt: DataLoaderAdaptor | dict[str, Any] = DataLoaderAdaptor(),
    ) -> Self:
        opt = DataLoaderAdaptor.parse(opt)
        iterator = opt.iterator_of(dataset)
        return cls(iterator, dataset.attrs)


class IteratorExecTypeError(AiowayError, TypeError): ...
