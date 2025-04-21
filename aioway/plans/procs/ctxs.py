# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import inspect
import typing
from collections.abc import Callable
from typing import ContextManager

from .procs import Proc


class CtxProc[C: Callable](Proc[C]):
    """
    Context manager processor, which wraps a function with a context manager.
    """

    @typing.override
    @typing.no_type_check
    def __call__(self, func: C) -> C:
        def wrapper(*args, **kwargs):
            with self.ctx(func):
                return func(*args, **kwargs)

        wrapper.__qualname__ = func.__qualname__
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper.__annotations__ = func.__annotations__
        wrapper.__signature__ = inspect.signature(func)

        return wrapper

    @abc.abstractmethod
    def ctx(self, func: C, /) -> ContextManager: ...
