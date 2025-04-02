# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing

from tensordict import TensorDict

from aioway.attrs import AttrSet
from aioway.blocks import Block
from aioway.errors import AiowayError

from .frames import Frame

__all__ = ["BlockFrame"]


@dcls.dataclass(frozen=True)
class BlockFrame(Frame):
    """
    A ``Frame`` backed by a ``Block``.
    This means that it is non-distributed, and volatile.
    """

    block: Block
    """
    The underlying data of the ``Frame``.
    """

    @typing.override
    def __len__(self):
        return len(self.block)

    def __getitem(self, idx):
        return self.block.data[idx]

    __getitem__ = __getitems__ = __getitem

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.block.attrs

    @property
    def device(self):
        return self.block.device


@dcls.dataclass(frozen=True)
class TensorDictFrame(Frame):
    """
    A ``Frame`` backed by a ``TensorDict``.
    This means that it is non-distributed, and volatile.
    """

    td: TensorDict
    """
    The underlying data of the ``Frame``.
    """

    attributes: AttrSet
    """
    The attribut sets for the current ``TensorDict`.
    """

    def __post_init__(self) -> None:
        if (block := Block(self.td)).attrs != self.attrs:
            raise TensorDictWrongAttrsError(f"{block.attrs=} != {self.attrs=}")

    @typing.override
    def __len__(self):
        return len(self.td)

    def __getitem(self, idx):
        return self.td[idx]

    __getitem__ = __getitems__ = __getitem

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        return self.attributes

    @property
    def device(self):
        return self.td.device


class TensorDictWrongAttrsError(AiowayError, ValueError): ...
