# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from collections.abc import Callable
from typing import ContextManager

from .opaque import OpaqueCall

__all__ = ["CtxCall"]


@dcls.dataclass(frozen=True)
class CtxCall[**P, T](OpaqueCall[P, T]):
    """
    Context manager processor, which wraps a function with a context manager.
    """

    @typing.override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        with self.ctx(self.func):
            result = self.func(*args, **kwargs)
        return result

    @abc.abstractmethod
    def ctx(self, func: Callable[P, T], /) -> ContextManager:
        """
        A context manager that takes in the callable.
        """

        ...
