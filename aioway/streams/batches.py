# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
from collections.abc import Iterator

import tensordict
from tensordict import TensorDict

from .streams import Stream

__all__ = ["BatchStream"]


@dcls.dataclass(frozen=True)
class BatchStream(Stream):
    """
    ``BatchStream`` consumes an input ``Stream`` and chunks it into batches of size ``batch``.
    """

    stream: Stream
    """
    The ``Stream`` instance to wrap.
    """

    batch: int
    """
    The batch size of the ``TensorDict``s to yield.
    """

    def __iter__(self) -> Iterator[TensorDict]:
        to_collate: list[TensorDict] = []

        for block in self.stream:
            to_collate_len = sum(map(len, to_collate))
            block_len = len(block)

            # Not enough length yet.
            if to_collate_len + block_len < self.batch:
                to_collate.append(block)
                continue

            keep = self.batch - to_collate_len

            preserve = block[:keep]

            to_collate.append(preserve)

            yield functools.reduce(
                lambda x, y: tensordict.cat([x, y]), to_collate[1:], to_collate[0]
            )

            to_collate = [] if keep == block_len else [block[keep:]]
