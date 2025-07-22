# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import difflib
import inspect
import typing
from abc import ABC
from collections import defaultdict as DefaultDict
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, Protocol

import structlog

from aioway.errors import AiowayError

__all__ = ["init_subclass", "of", "ClassRegistry", "GlobalRegistry"]

LOGGER = structlog.get_logger()


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
            # Allow abstract classes, which would not be initialized,
            # to not define keys, as factories are used to store leaf nodes.
            if inspect.isabstract(cls):
                return

            raise FactoryKeyError(
                f"Class: {cls} isn't given a key argument. Only valid for abstract classes."
            )

        factory[key] = cls

    return __init_subclass__


@dcls.dataclass(frozen=True)
class ClassRegistry(MutableMapping[str, type]):
    """
    Class registry for a given class (stored as attribute ``klass``).

    You will not be able to overwrite classes (2 classes with the same key),
    an error would be raised in that case.
    """

    klass: type
    """
    The class of which all registered classes must be a subclass.
    """

    registry: dict[str, type] = dcls.field(default_factory=dict)
    """
    The registry mapping from a key to a value.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.registry)

    @typing.override
    def __contains__(self, obj: Any) -> bool:
        return obj in self.registry

    @typing.override
    def __iter__(self):
        yield from self.registry

    @typing.override
    def __getitem__(self, key: str) -> type:
        if key not in self:
            self._raise_when_not_found(key)

        result = self.registry[key]
        self._raise_if_not_subclass(result)
        return result

    @typing.override
    def __setitem__(self, key: str, item: type) -> None:
        if key in self:
            raise FactoryKeyError(
                f"Trying to insert key: {key} and class: {item} "
                f"but key is already used by class: {self[key]}"
            )

        self._raise_if_not_subclass(item)
        self.registry[key] = item

    @typing.override
    def __delitem__(self, key: str) -> None:
        del self.registry[key]

    def _raise_if_not_subclass(self, t: type) -> None:
        if issubclass(t, self.klass):
            return

        raise FactoryKeyError(f"Type: {t} must be a subclass of class: {self.klass}.")

    def _raise_when_not_found(self, key):
        closest = difflib.get_close_matches(key, self)

        msg = f"{key=} not found in registry for class {self.klass}."

        if len(closest):
            candidates = ", ".join(f"'{close}'" for close in closest)
            msg = f"{msg}. Do you mean one of: [{candidates}]"

        raise FactoryKeyError(msg)


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


class FactoryKeyError(AiowayError, KeyError): ...
