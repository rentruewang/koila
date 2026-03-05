# Copyright (c) AIoWay Authors - All Rights Reserved

"The registry types using signatures as keys."

import typing
from collections import UserDict
from collections.abc import Callable
from typing import Any, TypeIs

from rich.table import Table

from .signatures import ParamList, Signature

__all__ = ["TypeCheckedDict", "PerTypeRegistry", "SignatureRegistry", "register"]


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
        return _reg_rich_table()

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


_REGISTRY = SignatureRegistry()


def register(signature: Signature | ParamList, key: str):
    "Register the callable based on their signature."

    def registrar[T: Callable](variant: T) -> T:
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


def _reg_rich_table() -> Table:
    "A rich table of operator vs signature."

    signatures = list(_REGISTRY.keys())
    table = Table(" ", *map(str, signatures))
    for op in sorted(_unique_ops()):
        items = [_REGISTRY[sign].get(op) for sign in signatures]
        table.add_row(op, *(i.__name__ if i else "-" for i in items))
    return table


def _unique_ops() -> set[str]:
    result: set[str] = set()
    for per_type in _REGISTRY.values():
        for op in per_type:
            result.add(op)
    return result
