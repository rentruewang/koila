# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import functools
import inspect
import typing
from abc import ABC
from collections.abc import Callable
from inspect import Signature
from typing import Self

from aioway import factories

__all__ = ["Proc", "OpaqueProc"]


@dcls.dataclass(frozen=True)
class Proc[**P, T](ABC):
    if typing.TYPE_CHECKING:

        def __init_subclass__(cls, *, key: str = ""):
            raise NotImplementedError

    else:
        __init_subclass__ = factories.init_subclass(lambda: Proc)

    @abc.abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Call the processor with the given arguments and keyword arguments.
        """

        ...


@dcls.dataclass(frozen=True)
class OpaqueProc[**P, T](Proc[P, T], key="OPAQUE"):
    """
    A processor that wraps a function, and calls then function for you.
    """

    func: Callable[P, T]
    """
    The function to be wrapped.
    """

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Call the ``func`` with the given arguments and keyword arguments.
        """

        return self.func(*args, **kwargs)

    @typing.no_type_check
    def __get__[E](self, instance: E | None, owner: type[E]):
        """
        The ``__get__`` method is used to make the processor callable as a method,
        and make it s.t. we can wrap methods and bound to ``self``.
        """

        _ = owner

        if instance is None:
            return self

        return functools.partial(self, instance)

    def __repr__(self) -> str:
        """
        The ``__repr__`` method is used to make the processor readable.
        """

        return f"{self.__class__.__name__}({self.func})"

    @property
    def __signature__(self) -> Signature:
        return inspect.signature(self.func)

    def with_func(self, func: Callable[P, T], /) -> Self:
        """
        Return a new processor with the given function.
        """

        return dcls.replace(self, func=func)
