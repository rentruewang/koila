# Copyright (c) AIoWay Authors - All Rights Reserved

"Turning on logging globally for a duration."

import contextlib as ctxl
import dataclasses as dcls
import logging
import typing
from collections import abc as cabc

from rich import logging as rlogging

from aioway._common import format_function

__all__ = ["enable_log", "enable_rich_log"]


@ctxl.contextmanager
def enable_log(level: str | int, /, *handlers: logging.Handler):
    """
    Enable logging for the duration of the block package wide.
    """

    logger = logging.getLogger("aioway")

    before_level = logger.level
    before_handlers = list(logger.handlers)

    for handler in handlers:
        logger.addHandler(handler)
        logger.setLevel(level)

    try:
        yield logger
    finally:
        logger.setLevel(before_level)
        logger.handlers = before_handlers


@ctxl.contextmanager
def enable_rich_log(level: str | int, /):
    """
    Enable logging for the duration of the block with rich handlers.
    """

    with enable_log(level, rlogging.RichHandler(show_path=False)) as logger:
        yield logger


type LoggingLevel = typing.Literal[
    "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", 0, 10, 20, 30, 40, 50
]
"The accepted logging levels. Same as `logging` library."


@dcls.dataclass(frozen=True)
class _CallAndLog[**P, T]:
    "Also logs the input and output of the function."

    log: cabc.Callable
    "The logger function."

    func: cabc.Callable[P, T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        "Call and log."

        formatted = format_function(self.func, *args, **kwargs)
        self.log("%s", formatted)
        result = self.func(*args, **kwargs)
        self.log("%s -> %s", formatted, result)
        return result


def _get_level_int(level: LoggingLevel):
    match level:
        case 0 | "NOTSET":
            return 0
        case 10 | "DEBUG":
            return 10
        case 20 | "INFO":
            return 20
        case 30 | "WARNING":
            return 30
        case 40 | "ERROR":
            return 40
        case 50 | "CRITICAL":
            return 50

    raise ValueError(level)
