# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import numpy as np
from tensordict import TensorDict

from .frames import Frame

__all__ = ["BatchFrame", "BatchListFrame"]


@dcls.dataclass(frozen=True)
class BatchFrame(Frame, key="BATCH"):
    """
    A ``Frame`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    td: TensorDict
    """
    The underlying data of the ``Frame``.
    """

    @typing.override
    def __len__(self):
        return len(self.td)

    def __getitem(self, idx):
        return self.td[idx]

    _getitem_int = _getitem_slice = _getitem_arr = _getitem_tensor = __getitem

    @property
    def device(self):
        return self.td.device


@dcls.dataclass(frozen=True)
class BatchListFrame(Frame, key="LIST"):
    """
    A ``Frame`` backed by a list of ``TensorDict``.

    The items in the lists are dynamically concatinated.
    Supports random access.

    This means that it is non-distributed, and volatile.

    """

    tds: list[TensorDict]
    """
    The underlying data of the ``Frame``.
    """

    @typing.override
    def __len__(self):
        return sum(self._lengths())

    def __getitem(self, idx):
        np.cumsum(self._lengths())

    __getitem__ = __getitems__ = __getitem

    def append(self, td: TensorDict) -> None:
        self.tds.append(td)

    def pop(self) -> TensorDict:
        return self.tds.pop()

    def _getitem_int(self, idx: int):
        lengths = self._lengths()
        cum_lengths = np.cumsum(lengths)
        part_idx = np.searchsorted(cum_lengths, idx)
        return self.tds[part_idx][lengths[part_idx] - idx]

    def _lengths(self) -> list[int]:
        return [len(td) for td in self.tds]

    @property
    def tds_arr(self):
        return np.array(self.tds)
