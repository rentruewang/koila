# Copyright (c) AIoWay Authors - All Rights Reserved

import copy
import functools
import typing
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import NamedTuple, Self

from tensordict import TensorDict
from tensordict._td import TensorDict

from ..tables import TorchListTable
from .execs import Exec

__all__ = ["IterExec", "UniqueExec", "SharedExec"]


class IterExec(Exec, ABC):
    """
    ``IterExec`` are ``Exec``s, but backed by ``Iterator``s.
    """

    def __init__(self, iterable: Iterable[TensorDict]) -> None:
        self._iter = iter(iterable)
        """
        The iterator to iterate over.
        """


@typing.final
class UniqueExec(IterExec):
    """
    The ``Exec`` that cannot be forked, and can call ``iter`` only once.
    This is because copying a running ``Iterator`` doesn't really make sense,
    and can lead to surprising behavior.
    """

    def __init__(self, iterable: Iterable[TensorDict]) -> None:
        super().__init__(iterable)

        self.__started = False
        """
        Flag to track whether or not the iterator has been started.
        """

    @typing.override
    def __next__(self) -> TensorDict:
        if not self.__started:
            raise TypeError("`UniqueExec` instance is not ``iter``-ed yet.")

        return next(self._iter)

    @typing.override
    def __iter__(self) -> Self:
        self.__started = True
        return self


class IndexBatch(NamedTuple):
    idx: int
    bat: TensorDict


@typing.final
class SharedExec(IterExec):
    """
    The ``Exec`` that will be shared, therefore, must cache to maintain consistent state.
    """

    def __init__(
        self, iterable: Iterable[TensorDict], frame: TorchListTable | None = None
    ) -> None:
        super().__init__(iterable)

        self._frame = frame or TorchListTable()
        """
        The caching frame storing the past batches.
        """
        self._next_cnt = 0
        """
        The number of times ``next(self)`` has been called.
        """

    @typing.override
    def __iter__(self):
        # This would make a shallow copy, s.t. ``idx`` is reset,
        # but the generator and ``Frame`` are not reset or copied.
        return copy.copy(self)

    @typing.override
    def __next__(self) -> TensorDict:
        batch = self._compute_next()
        self._next_cnt += 1
        return batch

    def _compute_next(self) -> TensorDict:
        """
        Compute the next item. Retrieve from ``frame`` if the item has been retrieved before,
        determined by the length of the ``TorchListFrame`` and ``self.idx``.
        """
        assert self.idx >= 0
        assert self.idx <= len(self._frame)

        # Must have seen this batch before.
        if self.idx < len(self._frame):
            return self._frame[self.idx]

        # Hitting unvisited territory.
        next_idx, batch = next(self._iter_batch)
        assert next_idx == self.idx, {
            "self.idx": self.idx,
            "next_idx": next_idx,
        }
        self._frame.append(batch)
        return batch

    @functools.cached_property
    def _iter_batch(self) -> Iterator[tuple[int, TensorDict]]:
        """
        The iterator with the iteration index are cached.
        This can only be used once, as the generator object would be cahced.
        """

        yield from enumerate(self._iter)

    @property
    def idx(self):
        return self._next_cnt
