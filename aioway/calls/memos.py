# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import typing

from .opaque import OpaqueCall

__all__ = ["MemoCall"]


@dcls.dataclass(frozen=True)
class MemoCall[**P, T](OpaqueCall[P, T]):
    """
    ``MemoProc`` is a processor that adds memoization to a function.
    """

    def __len__(self) -> int:
        return self._caller.cache_info().currsize

    @typing.override
    @typing.no_type_check
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Call the wrapped function with the given arguments and keyword arguments.
        """

        return self._caller(*args, **kwargs)

    @functools.cache
    def _caller(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Call the wrapped function with the given arguments and keyword arguments.
        """

        return self.func(*args, **kwargs)

    def reset(self) -> None:
        self._caller.cache_clear()
