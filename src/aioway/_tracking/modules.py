# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking high level modules."

import inspect
from collections.abc import Callable
from . import logging
import typing
import contextlib as ctxl
import dataclasses as dcls

if typing.TYPE_CHECKING:
    from aioway._ops import OpSign

# The global API logger.
LOGGER = logging.get_logger("__api__")


@dcls.dataclass(frozen=True)
class ModuleApiTracker:
    "The object for module API tracking."

    module: type
    "The high level module to track."

    name: str
    "The name of the operator."

    signature: OpSign
    "The signature of the operator. This is useful because an operator can be overloaded."

    @ctxl.contextmanager
    def __call__(self):
        module = self.module.__qualname__
        name = self.name
        sign = self.signature

        try:
            LOGGER.info("Enter %s::%s(%s)", module, name, sign)
            yield
        finally:
            LOGGER.info("Exit %s::%s(%s)", module, name, sign)

    def wrap[**P, T](self, function: Callable[P, T]) -> Callable[P, T]:

        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with self():
                return function(*args, **kwargs)

        wrapper.__name__ = function.__name__
        wrapper.__qualname__ = function.__qualname__
        wrapper.__doc__ = function.__doc__
        wrapper.__signature__ = inspect.signature(function)

        return wrapper
