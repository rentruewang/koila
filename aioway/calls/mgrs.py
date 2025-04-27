# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import logging
import typing
from collections.abc import Callable
from typing import Literal

from aioway.errors import AiowayError

from .opaque import OpaqueCall
from .rewrites import CallRewrite

__all__ = ["CallRewriteMgr"]

LOGGER = logging.getLogger(__name__)


type SupportedScope = Literal["static", "dynamic"]

SUPPORTED_SCOPE = typing.get_args(SupportedScope)


@typing.final
@dcls.dataclass(frozen=True)
class CallRewriteMgr[**P, T](CallRewrite[P, T]):
    """
    Compute the lifetime of the processors.
    """

    static: CallRewrite
    """
    Static processors, which are executed once when the function is defined.
    """

    dynamic: CallRewrite
    """
    Dynamic processors, which are executed every time the function is called.
    """

    @typing.override
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        return StaticDynamicCall(func=func, static=self.static, dynamic=self.dynamic)


@dcls.dataclass(frozen=True)
class StaticDynamicCall[**P, T](OpaqueCall[P, T]):
    func: Callable[P, T]
    static: CallRewrite
    dynamic: CallRewrite

    @typing.override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.dyn_call(*args, **kwargs)

    @functools.cached_property
    def static_call(self) -> Callable[P, T]:
        """
        The static processor property.

        Note:
            This property is cached because we do not need to recreate it again.
        """

        LOGGER.debug("Creating static processor for %s with %s", self.func, self.static)

        return self.static(self.func)

    @property
    def dyn_call(self) -> Callable[P, T]:
        """
        The dynamic processor property.

        Note:
            This property is not cached because we need to recreate it every time.
        """

        LOGGER.debug(
            "Creating dynamic processor for %s with %s", self.func, self.dynamic
        )

        return self.dynamic(self.static_call)


class ProcLifetimeScopeError(AiowayError, KeyError): ...
