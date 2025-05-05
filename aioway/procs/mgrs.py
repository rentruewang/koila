# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import logging
import typing
from typing import Literal

from aioway.errors import AiowayError

from .procs import OpaqueProc, Proc
from .rewrites import ProcRewrite

__all__ = ["ProcRewriteMgr"]

LOGGER = logging.getLogger(__name__)


type SupportedScope = Literal["static", "dynamic"]

SUPPORTED_SCOPE = typing.get_args(SupportedScope)


@typing.final
@dcls.dataclass(frozen=True)
class ProcRewriteMgr[**P, T](ProcRewrite[P, T]):
    """
    Compute the lifetime of the processors.
    """

    static: ProcRewrite
    """
    Static processors, which are executed once when the function is defined.
    """

    dynamic: ProcRewrite
    """
    Dynamic processors, which are executed every time the function is called.
    """

    @typing.override
    def __call__(self, func: Proc[P, T]) -> Proc[P, T]:
        return StaticDynamicProc(func=func, static=self.static, dynamic=self.dynamic)


@dcls.dataclass(frozen=True)
class StaticDynamicProc[**P, T](OpaqueProc[P, T], key="STATIC_DYNAMIC"):
    """
    A processor that combines static and dynamic processors.
    """

    static: ProcRewrite[P, T]
    """
    Static processor, which wraps a function with a static proxy function.
    """

    dynamic: ProcRewrite[P, T]
    """
    Dynamic processor, which wraps a function with a dynamic proxy function.
    """

    @typing.override
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.dyn_call(*args, **kwargs)

    @property
    def opaque_call(self) -> OpaqueProc[P, T]:
        """
        The opaque processor property.

        Note:
            This property is not cached because we need to recreate it every time.
        """

        LOGGER.debug("Creating opaque processor for %s", self.func)
        return OpaqueProc(self.func)

    @functools.cached_property
    def static_call(self) -> Proc[P, T]:
        """
        The static processor property.

        Note:
            This property is cached because we do not need to recreate it again.
        """

        LOGGER.debug("Creating static processor for %s with %s", self.func, self.static)

        return self.static(self.opaque_call)

    @property
    def dyn_call(self) -> Proc[P, T]:
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
