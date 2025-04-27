# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import logging
import typing
from abc import ABC
from collections.abc import Callable

__all__ = ["CallRewrite", "StaticCallRewrite", "ChainCallRewrite"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(frozen=True)
class CallRewrite[**P, T](ABC):
    """
    Proxy processor, which wraps a function with a proxy function.
    """

    @abc.abstractmethod
    def __call__(self, proc: Callable[P, T], /) -> Callable[P, T]:
        """
        Call the wrapped function with the given arguments and keyword arguments.
        """

        ...


@dcls.dataclass(frozen=True)
class StaticCallRewrite[**P, T](CallRewrite[P, T]):
    """
    Static processor, which wraps a function with a static proxy function.
    """

    transform: Callable[[Callable[P, T]], Callable[P, T]]

    @typing.override
    def __call__(self, func: Callable[P, T], /) -> Callable[P, T]:
        """
        A static proxy function that takes in the callable.
        """

        return self.transform(func)


@dcls.dataclass(frozen=True)
class ChainCallRewrite[**P, T](CallRewrite[P, T]):
    """
    Chain processor, which wraps a function with a chain of processors.
    """

    rewriters: tuple[CallRewrite[P, T], ...]

    @typing.override
    def __call__(self, func: Callable[P, T], /) -> Callable[P, T]:
        """
        A chain of processors that takes in the processor.
        """

        for rewrite in self.rewriters:
            func = rewrite(func)

        return func
