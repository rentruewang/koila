# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from collections.abc import Callable

from .opaque import OpaqueCall

__all__ = ["CallbackCall"]


@dcls.dataclass(frozen=True)
class CallbackCall[**P, T](OpaqueCall[P, T]):
    """
    A callback processor, which wraps a function with a callback function.
    """

    callback: Callable[[T], T] = dcls.field(repr=False)
    """
    The callback function to be called after the wrapped function.
    """

    @typing.override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Call the wrapped function with the given arguments and keyword arguments,
        and then call the callback function with the result.
        """

        result = self.func(*args, **kwargs)
        result = self.callback(result)
        return result
