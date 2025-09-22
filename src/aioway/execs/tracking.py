# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctx
import dataclasses as dcls
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Self

from aioway._errors import AiowayError

from .execs import Exec, ExecCtx

__all__ = ["ExecStack", "ExecProfiler"]


@dcls.dataclass(frozen=True)
class ExecStack(ExecCtx):
    """
    The stack context of execution.

    Corresponding to the current call stack, in ``Exec`` frames,
    where each ``Exec`` in the stack is in the next ``Exec``'s ``inputs()``.
    """

    stack: list[Exec]

    @ctx.contextmanager
    def __call__(self, frame: Exec):
        self.push(frame)

        try:
            yield
        finally:
            self.pop()

    def push(self, frame: Exec, /) -> None:
        "Push the current frame to the stack."
        self.push_ok(frame)
        self.stack.append(frame)

    def pop(self) -> Exec:
        "Pop the current frame to the stack."
        return self.stack.pop()

    def peek(self) -> Exec:
        "Peek the current frame to the stack."
        return self.stack[-1]

    def push_ok(self, frame: Exec) -> None:
        """
        Ensure that ``frame`` can be appended to ``self``.

        Since ``Exec`` exposes their input dependencies via ``inputs()``,
        the last item in the stack must be in ``frame.inputs()``.

        Args:
            frame: The current executing ``Exec``.
        """

        if self.peek() in frame.inputs():
            return

        raise StackError(
            f"Appending {frame=} to the current stack, "
            f"but {self.peek()=} is not in {frame.inputs()=}"
        )


@dcls.dataclass(frozen=True)
class ExecProfiler(ExecCtx):
    """
    A component by component profiler context.
    """

    profiler: Callable[[], AbstractContextManager]
    """
    The profiler to use.
    """

    @ctx.contextmanager
    def __call__(self, _: Exec):
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

        @ctx.contextmanager
        def profile():
            start = time.time()

            try:
                yield
            finally:
                end = time.time()
                print("Took", end - start, "seconds")

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

        @ctx.contextmanager
        def profile():
            with pyinstrument.profile():
                yield

        return cls(profile)


@dcls.dataclass(frozen=True)
class ExecPyInstrumentProfiler:
    @ctx.contextmanager
    def __call__(self):
        import pyinstrument

        with pyinstrument.profile():
            yield


class StackError(AiowayError, ValueError): ...
