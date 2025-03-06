# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import typing
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from aioway.blocks import Block

from .frames import Frame

__all__ = ["ReBatchFrame"]


@dcls.dataclass(frozen=True)
class ReBatchFrame(Frame):
    """
    Re-batches the input ``Frame``, and make it s.t. it uses the new batch size.
    """

    frame: Frame
    max_batch: int

    @typing.override
    def __len__(self):
        lengths = self._lengths()
        return sum(lengths)

    @typing.override
    def __getitem__(self, idx: int) -> Block:
        start = idx * self.max_batch
        stop = (idx + 1) * self.max_batch

        cum_len = self._cumulative_length()
        start_idx = np.searchsorted(cum_len, start)
        stop_idx = np.searchsorted(cum_len, stop)

        batches = [self.frame[i] for i in range(start_idx, stop_idx)]
        return functools.reduce(Block.chain, batches[1:], batches[0])

    @functools.cache
    def _lengths(self) -> Sequence[int]:
        return [len(block) for block in self.frame]

    def _cumulative_length(self) -> NDArray:
        lengths = self._lengths()
        return np.cumsum(lengths)
