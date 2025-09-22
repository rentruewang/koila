# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import contextlib as ctx
import logging
from abc import ABC
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack
from typing import Protocol

from tensordict import TensorDict

from aioway import _registries
from aioway._errors import AiowayError
from aioway.ops import BatchGen, Thunk

__all__ = ["Exec", "ExecCtx", "Execution"]

LOGGER = logging.getLogger(__name__)

type Execution = BatchGen


class ExecCtx(Protocol):
    """
    The execution contexts for the current
    """

    def __call__(self, exe: "Exec", /) -> AbstractContextManager: ...


class Exec(Iterable[TensorDict], ABC):
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

    def __init__(self, *ctxs: ExecCtx) -> None:
        self._ctxs = ctxs

    def __init_subclass__(cls, key: str):
        _registries.init_subclass(lambda: Exec)(cls, key=key)

    def __iter__(self) -> Execution:
        """
        The ``__iter__`` method launches a new ``Iterator`` to loop over the inputs.
        Every call is creates / rebuilds brand new computation.

        Returns:
            A stream of ``Block``s.
        """

        # Prior to ``__iter__`` cals, ensure that the dependencies look correct.
        exec_deps_thunks = {ex.thunk for ex in self.inputs()}
        thunk_deps = {*self.thunk.inputs()}
        if exec_deps_thunks != thunk_deps:
            raise ExecInputError(
                f"Dependencies {exec_deps_thunks} != Thunks's dependencies {thunk_deps}. This is a bug."
            )

        LOGGER.debug("Launching an iterator from %s", self)

        with self._enter_contexts():
            yield from self.iterate()

    @abc.abstractmethod
    def iterate(self) -> Execution:
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
        """

    @ctx.contextmanager
    def _enter_contexts(self):
        """
        Enter the contexts of ``self._ctxs``.
        """

        with ExitStack() as stack:
            for ec in self._ctxs:
                stack.enter_context(ec(self))

            # Should perform cleanup in case of failure as it's inside ``with``.
            yield


class ExecInputError(AiowayError, AssertionError): ...
