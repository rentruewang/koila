# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterable, Iterator

from tensordict import TensorDict

__all__ = ["Stream"]


class Stream(Iterable[TensorDict], ABC):
    """
    ``Stream`` represents a stream of heterogenious data being generated,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    It can be thought of as an ``Iterable`` of ``TensorDict``s,
    where computation happens eagerly, imperatively, and the result is yielded.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[TensorDict]:
        """
        Coroutine to iterate over the current ``Stream``.
        """

        ...
