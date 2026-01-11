# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Stream`` that records past histories and supports random access."

import abc
import dataclasses as dcls
import functools
import logging
import math
import typing
from abc import ABC
from collections.abc import Generator, Sequence
from typing import Self, override

from tensordict import TensorDict
from torch.utils.data import DataLoader, Sampler

from aioway.tables import Table

from .streams import Stream

__all__ = [
    "BoundedStream",
    "CacheStream",
    "ListStream",
    "TableStream",
    "TableStreamLoader",
]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass
class BoundedStream(Stream, ABC):
    """
    A stream with ``__len__`` and ``__getitem__``.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        "The number of batches saved in the current ``Stream``."

        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> TensorDict:
        """
        Get individual items. Does not support slice input.

        Args:
            idx: An integer. Must be in the range `[-len(self), len(self))`.

        Returns:
            The ``TensorDict`` batch.
        """


@dcls.dataclass
class CacheStream(BoundedStream):
    """
    Exhaust the input stream, store it into a cache for repeating access.
    """

    stream: Stream
    "The input stream."

    saved: list[TensorDict] = dcls.field(default_factory=list)
    "The cache for the input ``Stream``."

    @typing.override
    def __iter__(self) -> Self:
        return dcls.replace(self, stream=self.stream, saved=self.saved)

    @typing.override
    def __len__(self) -> int:
        return len(self.saved)

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        return self.saved[idx]

    @typing.override
    def _read(self) -> TensorDict:
        LOGGER.debug(
            "Executing `__iter__` for `CacheStream`. self.idx=%s, stream.idx=%s",
            self.idx,
            self.stream.idx,
        )

        if self.idx > self.stream.idx or self.idx > len(self.saved):
            raise AssertionError(
                "Invalid idx. Not synced properly with stream or cache."
            )

        # Try to get from ``self.saved`` first.
        if self.idx < len(self):
            return self[self.idx]

        # Now all the previous ones still must all have been saved.
        assert self.idx == len(self), f"{self.idx=} for {len(self)=}"

        # This shall raise ``StopIteration`` after done.
        # This may be fragile.
        item = next(self.stream)
        self.saved.append(item)
        return item

    @typing.override
    def _children(self) -> Generator[Stream]:
        yield self.stream

    @property
    @typing.override
    def size(self):
        return self.stream.size


@dcls.dataclass
class ListStream(BoundedStream):
    "A ``Stream`` backed by a list of ``TensorDict``."

    sequence: Sequence[TensorDict]
    "List of tensordicts."

    @typing.override
    def __len__(self) -> int:
        return self.size

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        return self.sequence[idx]

    @property
    @override
    def size(self) -> int:
        return len(self.sequence)

    @typing.override
    def _read(self) -> TensorDict:
        if self.idx < self.size:
            return self[self.idx]
        else:
            raise StopIteration

    @typing.override
    def _children(self) -> Generator[Stream]:
        return
        yield


@dcls.dataclass(frozen=True)
class TableStreamLoader:
    """
    The optoins for ``DataLoader`` on ``Table`` in ``TableStream``.
    """

    batch_size: int = 1
    "The batch size of individual batches."

    drop_last: bool = False
    "Whether to drop the last batch, which may have a different ``batch_size``."

    shuffle: bool = False
    "To shuffle or not."

    sampler: Sampler[int] | None = None
    "How to sample in case when want to shuffle."


@dcls.dataclass
class TableStream(Stream):
    """
    A ``Stream`` backed by a ``Table``.
    """

    table: Table
    "The underlying ``Table``."

    options: TableStreamLoader
    """
    The options passed directly to ``DataLoader``.
    """

    @typing.override
    def _read(self) -> TensorDict:
        try:
            return self._get_batch(self.idx)
        except IndexError as ie:
            raise StopIteration from ie

    @functools.cached_property
    @typing.no_type_check
    def _dataloader(self) -> DataLoader:
        # Note that ``__dict__`` of a dataclass is just the custom fields.
        return DataLoader(
            self.table,
            **self.options.__dict__,
            collate_fn=_identity,
        )

    @property
    @typing.override
    def size(self) -> int:
        return self._size

    @functools.cached_property
    def _size(self) -> int:
        batch_size = self.options.batch_size
        drop_last = self.options.drop_last
        rounding = math.floor if drop_last else math.ceil
        return rounding(len(self.table) / batch_size)

    def _get_batch(self, idx: int) -> TensorDict:
        batch_size = self.options.batch_size

        if not -self.size <= idx < self.size:
            raise IndexError(
                f"Index {idx=} out of bounds for stream {self.size=}, "
                f"backed by {self.table}."
            )

        idx %= self.size
        return self.table[idx : idx + batch_size]

    @typing.override
    def _children(self) -> Generator[Stream]:
        return
        yield


def _identity[T](item: T) -> T:
    return item
