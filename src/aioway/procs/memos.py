# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
import typing

from .procs import OpaqueProc

__all__ = ["MemoProc"]


@dcls.dataclass(frozen=True)
class MemoProc[**P, T](OpaqueProc[P, T], key="MEMO"):
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
