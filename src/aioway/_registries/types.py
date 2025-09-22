# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import logging
import typing
from abc import ABC
from collections import defaultdict as DefaultDict
from collections.abc import Callable, Mapping
from typing import Protocol

from .registries import Registry, RegistryKeyError

__all__ = ["init_subclass", "of", "ClassRegistry", "GlobalRegistry"]
LOGGER = logging.getLogger(__name__)


class RegistryInitSubclass[T: type[ABC]](Protocol):
    """
    The ``__init_subclass__`` method of the classes making use of a factory.
    """

    def __call__(self, cls: T, *, key: str = "") -> None: ...


def init_subclass[T: type[ABC]](base: Callable[[], T]) -> RegistryInitSubclass[T]:
    """
    Initialize the subclass, with a given base class.
    The subclass would need to specify a key (str), if not abstract,
    s.t. the subclass can later be retrived from a registry.


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

        factory = _GLOBAL_REGISTRY[base_class]

        if not key:
            raise RegistryKeyError(
                f"Class: {cls} isn't given a key argument. Only valid for abstract classes."
            )

        factory[key] = cls

    return __init_subclass__


@dcls.dataclass(frozen=True)
class _ClassMixin:
    klass: type
    """
    The class of which all registered classes must be a subclass.
    """


@dcls.dataclass(frozen=True, repr=False)
class ClassRegistry[T: type](Registry[T], _ClassMixin):
    """
    Class registry for a given class (stored as attribute ``klass``).
    """

    @typing.override
    def __getitem__(self, key: str) -> T:
        result = super().__getitem__(key)
        self._raise_if_not_subclass(result)
        return result

    @typing.override
    def __setitem__(self, key: str, item: T, /) -> None:
        super().__setitem__(key, item)
        self._raise_if_not_subclass(item)

    @typing.override
    def _raise_error_not_found(self, key: str):
        try:
            return super()._raise_error_not_found(key)
        except RegistryKeyError as rke:
            raise RegistryKeyError(
                f"Error occured during registry lookup for base class: {self.klass}"
            ) from rke

    def _raise_if_not_subclass(self, t: T, /) -> None:
        if issubclass(t, self.klass):
            return

        raise RegistryKeyError(f"Type: {t} must be a subclass of class: {self.klass}.")


@typing.no_type_check
def of[T: type[ABC]](cls: T) -> ClassRegistry:
    return _GLOBAL_REGISTRY[cls]


@dcls.dataclass(frozen=True)
class GlobalRegistry(Mapping[type, ClassRegistry]):
    factories: dict[type, dict[str, type]] = dcls.field(
        default_factory=lambda: DefaultDict(dict)
    )

    @typing.override
    def __contains__(self, key):
        return key in self.factories

    def __len__(self):
        return len(self.factories)

    @typing.override
    def __iter__(self):
        yield from self.factories

    @typing.override
    def __getitem__(self, klass: type) -> ClassRegistry:
        dicts = self.factories[klass]
        return ClassRegistry(klass=klass, registry=dicts)


_GLOBAL_REGISTRY = GlobalRegistry()
"""
A global registry to store all the registries.

Each registry is a ``ClassRegistry`` instance,
that handles looking up classes,
and suggesting a similar key registered in case of typo.
"""
