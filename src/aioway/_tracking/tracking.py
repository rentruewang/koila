# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import logging
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

__all__ = ["track", "track_function"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass
class FunctionLogger:
    function: str

    logger: Callable

    args: Sequence = ()

    kwargs: Mapping[str, Any] = MappingProxyType({})

    def params(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

        self.logger("%s", self._function_repr())

    def result(self, output: Any, /):
        self.logger("%s -> %s", self._function_repr(), output)

    def _function_repr(self) -> str:
        return _format_function(self.function, *self.args, **self.kwargs)


@ctxl.contextmanager
def track(name: str, logger: Callable[..., None]):
    yield FunctionLogger(name, logger=logger)


@ctxl.contextmanager
def track_function[**P, T](function: Callable[P, T], logger: Callable[..., None]):
    def tracked(*args: P.args, **kwargs: P.kwargs) -> T:
        with track(function.__name__, logger=logger) as tracker:
            tracker.params(*args, **kwargs)
            result = function(*args, **kwargs)
            tracker.result(result)
        return result

    tracked.__name__ = function.__name__
    tracked.__qualname__ = function.__qualname__
    tracked.__annotations__ = function.__annotations__
    tracked.__doc__ = function.__doc__

    yield tracked


def _format_function(func: str, *args: Any, **kwargs: Any) -> str:
    args_builder: list[str] = []

    # Add positional arguments.
    if args:
        args_builder.extend(map(str, args))

    # Add keyword arguments.
    if kwargs:
        args_builder.extend(f"{k}={v}" for k, v in kwargs.items())

    args_str = ", ".join(args_builder)
    return f"{func}({args_str})"
