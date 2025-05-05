# Copyright (c) RenChu Wang - All Rights Reserved

__all__ = ["LazyStr", "lazy_log"]

import dataclasses as dcls
import typing
from collections.abc import Callable
from logging import Logger
from types import MethodType
from typing import Any, Protocol

from aioway.errors import AiowayError

__all__ = ["LazyStr", "lazy_log"]


@dcls.dataclass(eq=False, frozen=True)
class LazyStr:
    """
    ``LazyStr`` is a way to construct a string,
    only if ``str`` or ``repr`` is called upon it.

    It is useful to avoid expensive computation in situations where
    logging level is too high to trigger (so no effect),
    but the string would need to be evaluated so there is wasted computation.
    """

    expr: Callable[[], str]
    """
    The expensive computation that would happen.
    """

    def __str__(self) -> str:
        return self()

    def __repr__(self) -> str:
        return self()

    def __eq__(self, other: object) -> bool:
        """
        Evaluate self, and compare with the string representation of other.
        """

        return str(self) == str(other)

    def __call__(self) -> str:
        """
        Implement ``__call__`` s.t. if users decide to
        pass this into the ``lazy_log`` function, it would still work.
        """
        return self.expr()


class LoggerMethod(Protocol):
    """
    The method compatible with the built in ``logging.Logger``.
    """

    __doc__: str
    __qualname__: str

    def __call__(self, msg: object, *args: object, **kwargs: Any) -> None: ...


class LazyLogger(Protocol):
    """
    A custom verion of the method where arguments are expensive.
    """

    __doc__: str
    __qualname__: str

    def __call__(
        self, msg: object, *args: Callable[[], object], **kwargs: Any
    ) -> None: ...


@typing.no_type_check
def lazy_log(log: LoggerMethod, /) -> LazyLogger:
    """
    Make a logger method lazy by accepting closures and evaluating them at runtime,
    if it turns out that the logging level is reached.

    This creates a ``LazyStr`` for each closure,
    which means that if the need arise (logging level is reached),
    the additional cost would be a lambda function creation, which is cheap.
    If the logging level is not reached, then a lot of computation would be spared,
    especially in the case of expensive and big objects, like dataclasses.

    Args:
        log: A method of ``logger.Logger``.

    Raises:
        LazyLoggerMethodError:
            If the given ``log`` method is not a bound method of ``Logger``.
    """

    if not callable(log):
        raise LazyLoggerMethodError(f"The logger method: {log} is not callable.")

    if not isinstance(log, MethodType):
        raise LazyLoggerMethodError(f"The function: {log} is not a bound method.")

    if not isinstance(log.__self__, Logger):
        raise LazyLoggerMethodError(
            f"The bound instance {log.__self__=} is not of instance ``Logger``."
        )

    def logger(msg: object, *args: Callable[[], Any], **kwargs: Any) -> None:
        lazy_args = [LazyStr(arg) for arg in args]
        log(msg, *lazy_args, **kwargs)

    logger.__doc__ = log.__doc__
    logger.__qualname__ = log.__qualname__

    return logger


class LazyLoggerMethodError(AiowayError, TypeError): ...
