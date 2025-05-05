# Copyright (c) RenChu Wang - All Rights Reserved

import inspect
import logging
import typing
from abc import ABC
from collections import defaultdict as DefaultDict
from collections.abc import Callable
from typing import Protocol

from aioway.errors import AiowayError

__all__ = ["init_subclass", "of"]

LOGGER = logging.getLogger(__name__)


class FactoryInitSubclass[T: type[ABC]](Protocol):
    """
    The ``__init_subclass__`` method of the classes making use of a factory.
    """

    def __call__(self, cls: T, *, key: str = "") -> None: ...


def init_subclass[T: type[ABC]](base: Callable[[], T]) -> FactoryInitSubclass[T]:
    """
    Initialize the subclass, with a given base class.

    Args:
        base:
            The base class to be used for the factory.
            This needs to be a lambda function because when ``__init_subclass__`` is defined,
            the class definition is not yet done, so Python would give ``NameError``.

    Returns:
        A callable function that can be used as a decorator to initialize the subclass.

    Examples:
        >>> class Base:
        ...     __init_subclass__ = init_subclass(lambda: Base)
        ...
        >>> class A(Base, key="a"): ...
        >>> class B(Base, key="b"): ...
        >>> fac = of(Base)
        >>> assert len(fac) == 2
        >>> assert fac.keys() == {"a", "b"}
        >>> assert fac["a"] is A
        >>> assert fac["b"] is B

    Fixme:
        Type hint issue about python/mypy#18987.
    """

    def __init_subclass__(cls, *, key: str = "") -> None:
        base_class = base()

        factory = _GLOBAL_FACTORIES[base_class]

        if not key:
            # Allow abstract classes, which would not be initialized,
            # to not define keys, as factories are used to store leaf nodes.
            if inspect.isabstract(cls):
                return

            raise FactoryKeyError(
                f"Class: {cls} isn't given a key argument. Only valid for abstract classes."
            )

        if key in factory:
            raise FactoryKeyError(
                f"Trying to insert key: {key} and class: {cls} "
                f"but key is already used by class: {factory[key]}"
            )

        factory[key] = cls

    return __init_subclass__


@typing.no_type_check
def of[T: type[ABC]](cls: T) -> dict[str, T]:
    return _GLOBAL_FACTORIES[cls]


_GLOBAL_FACTORIES: dict[type, dict[str, type]] = DefaultDict(dict)
"""
A global dictionary to store all the factories.
"""


class FactoryKeyError(AiowayError, KeyError): ...
