# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Iterator
from typing import Self

from tensordict import TensorDict
from torch.utils.data import DataLoader

from aioway.blocks import Block
from aioway.datatypes import AttrSet
from aioway.errors import AiowayError

from .execs import Exec

__all__ = ["IteratorExec"]


@dcls.dataclass(frozen=True)
class IteratorExec(Exec):
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
    def attrs(self):
        return self._attrs

    @classmethod
    def data_loader(cls, loader: DataLoader, attrs: AttrSet) -> Self:
        def load_from_dl():
            for batch in loader:
                if not isinstance(batch, TensorDict):
                    raise IteratorExecTypeError(
                        f"`DataLoader` should yield instances of `TensorDict`. Got {type(batch)=}"
                    )

                yield Block(batch)

        return cls(load_from_dl(), attrs)


class IteratorExecTypeError(AiowayError, TypeError): ...
