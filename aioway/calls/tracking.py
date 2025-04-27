# Copyright (c) RenChu Wang - All Rights Reserved

import contextlib
import dataclasses as dcls
import logging
import typing
from collections.abc import Callable

from .ctxs import CtxCall

__all__ = ["StackCall", "LoggingProc"]

LOGGER = logging.getLogger(__name__)


@typing.final
@dcls.dataclass(frozen=True)
class StackCall[**P, T](CtxCall[P, T]):
    """
    The callstack for an ``Exec``.
    It is used to store the current state of the ``Exec``.
    """

    stack: list[Callable[P, T]] = dcls.field(default_factory=list)
    """
    The stack of ``Exec``s.
    """

    def __len__(self) -> int:
        return len(self.stack)

    def __getitem__(self, index: int) -> Callable[P, T]:
        return self.stack[index]

    @typing.override
    @contextlib.contextmanager
    def ctx(self, func: Callable[P, T]):
        """
        Create a new scope manager with the given ``Exec``.
        """

        self.__append(func)

        yield self

        popped = self.__pop()

        assert (
            popped is func
        ), "Instance popped from stack: {} should be exec instead: {}".format(
            popped, func
        )

    def __append(self, exec: Callable[P, T]) -> None:
        """
        Append an ``Callable[P, T]`` to the context.
        """

        self.stack.append(exec)

    def __pop(self) -> Callable[P, T]:
        """
        Pop an ``Callable[P, T]`` from the context.
        """

        return self.stack.pop()

    @property
    def top(self) -> Callable[P, T]:
        """
        Get the top ``Callable[P, T]`` from the context.
        """

        return self.stack[-1]

    @property
    def root(self) -> Callable[P, T]:
        """
        Get the root ``Callable[P, T]`` from the context.
        """

        return self.stack[0]


class LoggingProc[**P, T](CtxCall):
    """
    A logging processor that logs the function call and its arguments.
    """

    @contextlib.contextmanager
    def ctx(self, func: Callable[P, T]):
        """
        A context manager that logs the function call and its arguments.
        """

        LOGGER.debug(f"Preparing to call %s", func)

        yield func

        LOGGER.debug(f"Finished calling %s", func)
