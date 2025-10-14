# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from ..tables import Table
from .plans import Plan0

__all__ = ["FramePlan"]


@dcls.dataclass(frozen=True)
class FramePlan(Plan0):
    """
    An ``Plan`` that wraps a ``Frame`` and a ``DataLoader``.
    """

    dataset: "Table" = dcls.field(repr=False)
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
