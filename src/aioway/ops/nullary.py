# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from aioway.io import Frame

from .ops import Op0

__all__ = ["FrameOp"]


@dcls.dataclass(frozen=True)
class FrameOp(Op0, key="FRAME"):
    """
    An ``Op`` that wraps a ``Frame`` and a ``DataLoader``.
    """

    dataset: "Frame" = dcls.field(repr=False)
    """
    The backing ``Frame``, stored in order to reset.
    """

    def __hash__(self) -> int:
        """
        The hash function for ``Frame`` op.

        Note:
            As of now, using ``id`` s.t. it will work for cases,
            where only if the ``Frame`` is the same instance, for safety.
        """

        return id(self)

    @typing.override
    def stream(self):
        yield from self.dataset
