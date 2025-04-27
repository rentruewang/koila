# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
import inspect
import typing
from abc import ABC
from collections.abc import Callable
from inspect import Signature
from typing import Self

__all__ = ["OpaqueCall"]


@dcls.dataclass(frozen=True)
class OpaqueCall[**P, T](ABC):
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
