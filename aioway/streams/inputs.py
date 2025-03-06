# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator

from aioway.blocks import Block
from aioway.schemas import TableSchema

from .streams import Stream

__all__ = ["IteratorStream"]


@dcls.dataclass(frozen=True)
class IteratorStream(Stream):
    """
    ``IteratorStream`` is an adaptor that converts
    from an ``Iterator`` of ``Block``s into a ``Stream``.
    """

    iterator: Iterator[Block]
    """
    The ``Iterator`` that produces ``Block``s.
    """

    _schema: TableSchema = dcls.field(repr=False)
    """
    The schema of the ``Stream``.

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
    def schema(self):
        return self._schema
