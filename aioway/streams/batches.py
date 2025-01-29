# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
from collections.abc import Iterator

from aioway.blocks import Block

from .streams import Stream


@dcls.dataclass(frozen=True)
class BatchStream(Stream):
    stream: Stream
    batch: int

    def __iter__(self) -> Iterator[Block]:
        to_collate: list[Block] = []

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

            yield functools.reduce(Block.zip, to_collate[1:], to_collate[0])

            to_collate = [] if keep == block_len else [block[keep:]]
