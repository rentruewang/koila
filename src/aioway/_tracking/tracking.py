# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl
import dataclasses as dcls
import logging
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

__all__ = ["track"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass
class FunctionTracker:
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
    yield FunctionTracker(name, logger=logger)


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
