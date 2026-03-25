# Copyright (c) AIoWay Authors - All Rights Reserved

"Turning on logging globally for a duration."

import contextlib as ctxl
import dataclasses as dcls
import logging
from collections.abc import Callable
from logging import Handler
from typing import Any, Literal

from rich.logging import RichHandler

__all__ = ["enable_log", "enable_rich_log", "Logger", "get_logger"]


@ctxl.contextmanager
def enable_log(level: str | int, /, *handlers: Handler):
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

    with enable_log(level, RichHandler(show_path=False)) as logger:
        yield logger


type LoggingLevel = Literal[
    "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", 0, 10, 20, 30, 40, 50
]
"The accepted logging levels. Same as `logging` library."


@dcls.dataclass(frozen=True)
class Logger:
    "The logger class for `aioway`. Tries to mimic the API for `logging.Logger`."

    module: str
    "The module name passed to the logger."

    def log(self, level: LoggingLevel, msg: str, *args, **kwargs) -> None:
        level_int = _get_level_int(level)
        self._logger.log(level_int, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def is_enabled_for(self, level: LoggingLevel) -> bool:
        "Wrapper for `logging.Logger.isEnabledFor`."
        return self._logger.isEnabledFor(_get_level_int(level))

    def function(self, level: LoggingLevel, /):
        """
        Logs an entire function according to the `level` given.

        Replaces the function with a closure that uses the module's logger for logging purposes.
        This means that the settings changes to the logging would reflect in the new function.
        """

        level_int = _get_level_int(level)
        return self.__log_inputs_outputs(level_int)

    def __log_inputs_outputs(self, level: int):
        logger = self._logger

        def decorator[**P, T](f: Callable[P, T]) -> Callable[P, T]:
            # The actual wrapper wrapping the function.
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Here we assume that logging level is not changed for the duration for the module.
                enabled = logger.isEnabledFor(level)
                log = lambda *args, **kwargs: logger.log(level, *args, **kwargs)

                # If enabled, use a logger function.
                function = _CallAndLog(log, f) if enabled else f
                return function(*args, **kwargs)

            wrapper.__name__ = f.__name__
            wrapper.__qualname__ = f.__qualname__
            wrapper.__doc__ = f.__doc__

            return wrapper

        return decorator

    @property
    def _logger(self):
        return logging.getLogger(self.module)

    @property
    def level(self) -> int:
        return self._logger.level


def get_logger(module: str, /) -> Logger:
    "A replacement for `logging.getLogger`."

    return Logger(module)


@dcls.dataclass(frozen=True)
class _CallAndLog[**P, T]:
    "Also logs the input and output of the function."

    log: Callable
    "The logger function."

    func: Callable[P, T]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        "Call and log."

        formatted = _format_function(self.func, *args, **kwargs)
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


def _format_function(func: Callable, *args: Any, **kwargs: Any) -> str:
    "Format the function into readable string, mimicking signature in python."

    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(f"{arg!r}" for arg in args)

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k!s}={v!r}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    func_str = func.__qualname__
    return f"{func_str}({args_str})"
