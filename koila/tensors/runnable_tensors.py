from typing import Callable, NamedTuple


class BatchedPair(NamedTuple):
    batch: int
    no_batch: int


class BatchInfo(NamedTuple):
    index: int
    value: int

    def map(self, func: Callable[[int], int]) -> BatchInfo:
        index = func(self.index)
        return BatchInfo(index, self.value)
