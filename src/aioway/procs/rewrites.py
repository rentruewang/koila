# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import Callable

import structlog

from .procs import Proc

__all__ = ["ProcRewrite", "OpaqueProcRewrite", "ChainProcRewrite"]

LOGGER = structlog.get_logger()


@dcls.dataclass(frozen=True)
class ProcRewrite[**P, T](ABC):
    """
    Proxy processor, which wraps a function with a proxy function.
    """

    @abc.abstractmethod
    def __call__(self, proc: Proc[P, T], /) -> Proc[P, T]:
        """
        Call the wrapped function with the given arguments and keyword arguments.
        """

        ...


@dcls.dataclass(frozen=True)
class OpaqueProcRewrite[**P, T](ProcRewrite[P, T]):
    """
    Static processor, which wraps a function with a static proxy function.
    """

    rewrite: Callable[[Proc[P, T]], Proc[P, T]]
    """
    The static proxy function to be called after the wrapped function.
    """

    @typing.override
    def __call__(self, func: Proc[P, T], /) -> Proc[P, T]:
        """
        A static proxy function that takes in the callable.
        """

        return self.rewrite(func)


@dcls.dataclass(frozen=True)
class ChainProcRewrite[**P, T](ProcRewrite[P, T]):
    """
    Chain processor, which wraps a function with a chain of processors.
    """

    rewriters: tuple[ProcRewrite[P, T], ...]
    """
    A chain of processors that takes in the callable.
    """

    @typing.override
    def __call__(self, func: Proc[P, T], /) -> Proc[P, T]:
        """
        A chain of processors that takes in the processor.
        """

        for rewrite in self.rewriters:
            func = rewrite(func)

        return func
