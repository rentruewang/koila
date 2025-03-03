# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from collections.abc import Iterator, Sequence
from typing import Protocol

from tensordict import TensorDict

__all__ = ["BatchStream", "BatchList"]


class BatchStream(Protocol):
    """
    ``BatchStream`` represents a possibly unbounded stream of ``TensorDict``s.
    """

    @abc.abstractmethod
    def keys(self) -> Sequence[str]: ...

    @abc.abstractmethod
    def iterator(self) -> Iterator[TensorDict]: ...


class BatchList(BatchStream, Protocol):
    """
    ``BatchList`` represents a bounded sequence of ``TensorDict``s.
    """

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def batch(self, idx: int) -> TensorDict: ...

    def iterator(self) -> Iterator[TensorDict]:
        for i in range(self.count()):
            yield self.batch(i)


@dcls.dataclass(frozen=True)
class IterBatchStream(BatchStream):
    columns: Sequence[str]
    generator: Iterator[TensorDict]

    def keys(self) -> Sequence[str]:
        return self.columns

    def iterator(self) -> Iterator[TensorDict]:
        return self.generator


@dcls.dataclass(frozen=True)
class SeqBatchList(BatchList):
    columns: Sequence[str]
    sequence: Sequence[TensorDict]

    def keys(self) -> Sequence[str]:
        return self.columns

    def count(self) -> int:
        return len(self.sequence)

    def batch(self, idx: int) -> TensorDict:
        return self.sequence[idx]
