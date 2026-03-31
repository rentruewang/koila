# Copyright (c) AIoWay Authors - All Rights Reserved

"Tracking high level modules."

import contextlib as ctxl
import dataclasses as dcls
import typing
from collections import abc as cabc

from . import logging

if typing.TYPE_CHECKING:
    from aioway import _signs

__all__ = ["get_tracker"]


# The global API logger.
LOGGER = logging.get_logger("aioway.__api__")


def _logging_enter(info: ModuleMethodInfo) -> None:
    LOGGER.info("Enter %s.%s%s", info.module.__qualname__, info.name, info.signature)


def _logging_exit(info: ModuleMethodInfo) -> None:
    LOGGER.info("Exit %s.%s%s", info.module.__qualname__, info.name, info.signature)


def get_tracker(
    module: cabc.Callable[[], type],
    enter: cabc.Callable[[ModuleMethodInfo], None] = _logging_enter,
    exit: cabc.Callable[[ModuleMethodInfo], None] = _logging_exit,
):
    return ModuleApiTracker(module=module, enter=enter, exit=exit)


@dcls.dataclass(frozen=True)
class ModuleApiTracker:
    "The object for module API tracking."

    module: cabc.Callable[[], type]
    """
    The function to high level module to track.
    The reason this is a callable is to allow lambdas,
    s.t. the tracker can be placed before the class definition, which can be more elegant.
    """

    enter: cabc.Callable[[ModuleMethodInfo], None] = _logging_enter
    "The function to call before entering."

    exit: cabc.Callable[[ModuleMethodInfo], None] = _logging_exit
    "The function to call before exiting."

    @ctxl.contextmanager
    def __call__(self, name: str, signature: _signs.Signature):
        """
        Track the module's operator (name and signature of the operator).

        Args:
            name: The name of the operator.
            signature: The signature of the operator. This is useful because an operator can be overloaded.
        """

        info = ModuleMethodInfo(module=self.module(), name=name, signature=signature)

        try:
            self.enter(info)
            yield
        finally:
            self.exit(info)

    def wrap[**P, T](self, name: str, signature: _signs.Signature):

        def decorator(function: cabc.Callable[P, T]) -> cabc.Callable[P, T]:
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                with self(name=name, signature=signature):
                    return function(*args, **kwargs)

            wrapper.__name__ = function.__name__
            wrapper.__qualname__ = function.__qualname__
            wrapper.__doc__ = function.__doc__

            return wrapper

        return decorator


@dcls.dataclass(frozen=True)
class ModuleMethodInfo:
    module: type
    name: str
    signature: _signs.Signature
