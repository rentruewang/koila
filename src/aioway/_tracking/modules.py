# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking high level modules."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections.abc import Callable

from aioway._ops.signs import OpSign

from . import logging

if typing.TYPE_CHECKING:
    from aioway._ops import OpSign

__all__ = ["ModuleApiTracker"]

# The global API logger.
LOGGER = logging.get_logger("aioway.__api__")


@dcls.dataclass(frozen=True)
class ModuleApiTracker:
    "The object for module API tracking."

    module: Callable[[], type]
    """
    The function to high level module to track.
    The reason this is a callable is to allow lambdas,
    s.t. the tracker can be placed before the class definition, which can be more elegant.
    """

    @ctxl.contextmanager
    def __call__(self, name: str, signature: OpSign):
        """
        Track the module's operator (name and signature of the operator).

        Args:
            name: The name of the operator.
            signature: The signature of the operator. This is useful because an operator can be overloaded.
        """

        module = self.module().__qualname__

        try:
            LOGGER.info("Enter %s.%s%s", module, name, signature)
            yield
        finally:
            LOGGER.info("Exit %s.%s%s", module, name, signature)

    def wrap[**P, T](self, name: str, signature: OpSign):

        def decorator(function: Callable[P, T]) -> Callable[P, T]:
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                with self(name=name, signature=signature):
                    return function(*args, **kwargs)

            wrapper.__name__ = function.__name__
            wrapper.__qualname__ = function.__qualname__
            wrapper.__doc__ = function.__doc__

            return wrapper

        return decorator
