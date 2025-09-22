# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
from abc import ABC
from collections.abc import Iterable

from tensordict import TensorDict

from aioway import _registries
from aioway.ops import BatchGen, Thunk

__all__ = ["Exec", "Execution"]

LOGGER = logging.getLogger(__name__)

type Execution = BatchGen


class Exec(Iterable[TensorDict], ABC):
    """
    ``Exec`` is the graph / symbolic representation of an execution.
    Responsible for launching an ``BatchGen`` everytime ``__iter__`` is called.

    An execution itself does not store state,
    but rather launches an iterator / cursor to iterate over the data.
    This design allows users to write genertors (``__iter__`` funciton),
    rather than iterators (``__next__`` function) with state management.
    """

    def __init_subclass__(cls, key: str):
        _registries.init_subclass(lambda: Exec)(cls, key=key)

    def __iter__(self) -> Execution:
        """
        The ``__iter__`` method launches a new ``Iterator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Returns:
            A stream of ``Block``s.
        """

        LOGGER.debug("Launching an iterator from %s", self)
        yield from self.iterate()

    @abc.abstractmethod
    def iterate(self) -> Execution:
        """
        The implementation for ``__iter__``.
        """

    @property
    @abc.abstractmethod
    def thunk(self) -> Thunk:
        """
        The ``Thunk`` that this ``Exec``'s execution represents.

        Since every ``Exec`` is conceptually executed on a ``Thunk``,
        this attribute shows dependency, and can be used in comparison.
        """
