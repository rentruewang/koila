# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
import typing
from abc import ABC
from collections.abc import Iterator

from aioway._errors import AiowayError
from aioway.ops import BatchGen
from aioway.thunks import Thunk

if typing.TYPE_CHECKING:
    from .contexts import ExecCtx

__all__ = ["Exec"]


LOGGER = logging.getLogger(__name__)


class Exec(ABC):
    """
    ``Exec`` is the graph / symbolic representation of an execution.
    Responsible for launching an ``BatchGen`` everytime ``__iter__`` is called.

    It is responsible to execute a ``Thunk``,
    and has a 1 to 1 relationship with ``Thunk``s,
    where each ``Thunk`` would require an ``Exec`` to run.

    An execution itself does not store state,
    but rather launches an iterator / cursor to iterate over the data.
    This design allows users to write genertors (``__iter__`` funciton),
    rather than iterators (``__next__`` function) with state management.
    """

    def __init__(self) -> None:
        from .contexts import ExecCtx, ExecNullCtx

        self._ctx: ExecCtx = ExecNullCtx()

    def __iter__(self) -> BatchGen:
        """
        The ``__iter__`` method launches a new ``Iterator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Returns:
            A stream of ``Block``s.
        """

        LOGGER.debug("Launching an iterator from %s", self)

        with self.ctx(self):
            yield from self.iter()

    @abc.abstractmethod
    def iter(self) -> BatchGen:
        """
        The implementation for ``__iter__``.
        """

    @abc.abstractmethod
    def inputs(self) -> Iterator["Exec"]:
        """
        The dependencies (``Exec``) instances to the current ``Exec``.
        """

    @property
    @abc.abstractmethod
    def thunk(self) -> Thunk:
        """
        The ``Thunk`` that this ``Exec``'s execution represents.

        Since every ``Exec`` is conceptually executed on a ``Thunk``,
        this attribute shows dependency, and can be used in comparison.

        Returns:
            A ``Thunk``.
        """

    @property
    def ctx(self) -> "ExecCtx":
        return self._ctx

    @ctx.setter
    def ctx(self, ctx: "ExecCtx") -> None:
        self._ctx = ctx

        # Recursively update the children.
        for ipt in self.inputs():
            ipt.ctx = ctx


class ExecInputError(AiowayError, AssertionError): ...
