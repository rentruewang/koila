# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import time
import typing
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol, Self

from .execs import Exec

__all__ = ["ExecCtx", "ExecNullCtx", "ExecIceCream", "ExecBreakPoint", "ExecProfiler"]


class ExecCtx(Protocol):
    """
    The execution contexts for the current ``Exec``.

    This allows easily adding _non-invasive_ operations to the pipeline,
    such as logging, tracking etc.
    Invasive operations should not make use of this interface,
    and this is reflected in the interface design,
    that the ``contextmanger`` always ``yield None``.
    """

    def __call__(self, exe: Exec, /) -> AbstractContextManager[None]: ...


class ExecNullCtx(ExecCtx):
    @ctxl.contextmanager
    def __call__(self, _: Exec):
        yield


class ExecIceCream(ExecCtx):
    """
    Using ``icecream`` to aid debugging entering and exiting the executor.
    """

    @typing.override
    @ctxl.contextmanager
    def __call__(self, exe: Exec):
        import icecream

        icecream.ic("enter", self, exe)
        try:
            yield
        finally:
            icecream.ic("exit", self, exe)


class ExecBreakPoint(ExecCtx):
    @typing.override
    @ctxl.contextmanager
    def __call__(self, exe: Exec):
        breakpoint()
        yield


@dcls.dataclass(frozen=True)
class ExecProfiler(ExecCtx):
    """
    A component by component profiler context.
    """

    profiler: Callable[[], AbstractContextManager]
    """
    The profiler to use.
    """

    @typing.override
    @ctxl.contextmanager
    def __call__(self, _: "Exec"):
        with self.profiler():
            yield

    @classmethod
    def time(cls) -> Self:
        """
        A profiler based on ``time.time()``.

        Todo:
            Store the result of the states somewhere,
            not just logging to the console.
        """

        import icecream

        @ctxl.contextmanager
        def profile():
            start = time.time()

            try:
                yield
            finally:
                end = time.time()
                icecream.ic("Took", end - start, "seconds")

        return cls(profile)

    @classmethod
    def pyinstrument(cls) -> Self:
        """
        A profiler based on ``pyinstrument``.
        By default, this logs the outputs to the console.

        Todo:
            Add some filters s.t. the entire screen isn't polluted.
        """

        import pyinstrument

        @ctxl.contextmanager
        def profile():
            with pyinstrument.profile():
                yield

        return cls(profile)
