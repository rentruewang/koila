# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import logging
from abc import ABC

from aioway import registries
from aioway.ops import BatchGen, Thunk

__all__ = ["Exec", "execute"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class Exec(ABC):
    """
    ``Exec`` is the graph / symbolic representation of an execution.

    An execution itself does not store state,
    but rather launches an iterator / cursor to iterate over the data.
    This design allows users to write genertors (``__iter__`` funciton),
    rather than iterators (``__next__`` function) with state management.

    Todo:
        Currently ``Exec`` uses the lazy evaluation approach.
        To make it flexible and useful, make multiple implementations of ``Exec``.
    """

    thunk: Thunk
    """
    The thunk for which ``Exec`` is responsible for executing.
    """

    def __init_subclass__(cls, key: str):
        registries.init_subclass(lambda: Exec)(cls, key=key)

    @abc.abstractmethod
    def __iter__(self) -> BatchGen:
        """
        The ``__iter__`` method launches a new ``Iterator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Returns:
            A stream of ``Block``s.

        Note:
            Perhaps implement STG (#77).
        """

        ...

    @classmethod
    def from_thunk(cls, thunk: Thunk, /):
        return cls(thunk=thunk)


def execute(thunk: Thunk, strategy: str) -> Exec:
    registry = registries.of(Exec)
    return registry[strategy](thunk)
