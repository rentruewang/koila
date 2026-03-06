# Copyright (c) AIoWay Authors - All Rights Reserved

"The registry types using signatures as keys."

import typing
from collections import UserDict
from collections.abc import Callable
from typing import Any, Self, TypeIs

from rich.table import Table

from .signatures import ParamList, Signature

__all__ = [
    "TypeCheckedDict",
    "PerTypeRegistry",
    "SignatureRegistry",
    "register",
    "default_registry",
    "registry_for",
]


class TypeCheckedDict[K, V](UserDict[K, V]):
    """
    A dictionary that performs type checks on both keys and values.

    Subclasses should define how the type checks are performed.
    """

    def __getitem__(self, key: K) -> V:
        self.__validate_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key: K, item: V) -> None:
        self.__validate_key(key)
        self.__validate_val(item)
        return super().__setitem__(key, item)

    def __delitem__(self, key: K) -> None:
        self.__validate_key(key)
        super().__delitem__(key)

    @classmethod
    def _is_key_type(cls, key) -> TypeIs[K]:
        "This function returns true if subtype's key has the right type."

        return True

    @classmethod
    def _is_val_type(cls, val) -> TypeIs[V]:
        "This function returns true if subtype's value has the right type."

        return True

    @classmethod
    def __validate_key(cls, key):
        if not cls._is_key_type(key):
            raise TypeError(f"Key type: {type(key)} is not supported.")

    @classmethod
    def __validate_val(cls, value):
        if not cls._is_val_type(value):
            raise TypeError(f"Value type: {type(value)} is not supported.")


class PerTypeRegistry(TypeCheckedDict[str, Callable]):
    @classmethod
    @typing.override
    def _is_key_type(cls, key) -> TypeIs[str]:
        return isinstance(key, str)

    @classmethod
    @typing.override
    def _is_val_type(cls, val) -> TypeIs[Callable[..., Any]]:
        return callable(val)


class SignatureRegistry(TypeCheckedDict[ParamList, PerTypeRegistry]):
    """
    The global registry that is based on signatures.
    """

    def __rich__(self):
        return _reg_rich_table(self)

    @typing.override
    def __getitem__(self, key: ParamList) -> PerTypeRegistry:
        # DefaultDict behavior.
        if key not in self:
            self[key] = PerTypeRegistry()

        return super().__getitem__(key)

    @classmethod
    @typing.override
    def _is_key_type(cls, key) -> TypeIs[ParamList]:
        return isinstance(key, ParamList)

    @classmethod
    @typing.override
    def _is_val_type(cls, val) -> TypeIs[PerTypeRegistry]:
        return isinstance(val, PerTypeRegistry)

    @property
    def signatures(self):
        return set(self.keys())

    @property
    def ops(self):
        return _unique_ops(self)

    def select(self, *signatures: ParamList) -> Self:
        "Only view the types of selected signatures."

        return type(self)({sig: self[sig] for sig in signatures})


_REGISTRY = SignatureRegistry()


def default_registry():
    return _REGISTRY


def register(signature: Signature | ParamList, /, *keys: str):
    "Register the callable based on their signature."

    def registrar[T: Callable](variant: T) -> T:
        for key in keys:
            registry_for(signature)[key] = variant
        return variant

    return registrar


def registry_for(signature: Signature | ParamList, /) -> PerTypeRegistry:
    "Get the registry for a given signature."

    return _REGISTRY[_registry_key(signature)]


def _registry_key(signature: Signature | ParamList, /) -> ParamList:
    "Convert the signature into the key for registry."

    match signature:
        case Signature(params):
            return params
        case ParamList():
            return signature
        case _:
            raise TypeError(type(signature))


def _reg_rich_table(registry: SignatureRegistry, /) -> Table:
    "A rich table of operator vs signature."

    signatures = list(registry.keys())
    table = Table(" ", *map(str, signatures))
    for op in sorted(_unique_ops(registry)):
        items = [registry[sign].get(op) for sign in signatures]
        table.add_row(op, *(i.__name__ if i else "-" for i in items))
    return table


def _unique_ops(registry: SignatureRegistry, /) -> set[str]:
    result: set[str] = set()
    for per_type in registry.values():
        for op in per_type:
            result.add(op)
    return result
