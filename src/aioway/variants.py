# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import typing
from abc import ABC
from collections import UserDict
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, TypeIs

__all__ = ["register_1", "register_2", "find", "find_1", "find_2"]

type _Argc = Literal[1, 2, -1]


class RegKey(NamedTuple):
    "The registry key in tuple form."

    typ: type
    """
    The type of the operator. Corresponds to the columns of the specialization table.
    """

    op: str
    """
    The operator's name. Corresponds to the row of the specialization table.
    """


class _TypeCheckDict[K, V](UserDict[K, V], ABC):

    def __getitem__(self, key: K):
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
    @abc.abstractmethod
    def _is_key_type(cls, key) -> TypeIs[K]: ...

    @classmethod
    @abc.abstractmethod
    def _is_val_type(cls, val) -> TypeIs[V]: ...

    @classmethod
    def __validate_key(cls, key):
        if not cls._is_key_type(key):
            raise TypeError(key)

    @classmethod
    def __validate_val(cls, value):
        if not cls._is_val_type(value):
            raise ValueError(value)


class _Registry2D(_TypeCheckDict[RegKey, Callable[..., Any]]):
    @classmethod
    @typing.override
    def _is_key_type(cls, key) -> TypeIs[RegKey]:
        return isinstance(key, RegKey)

    @classmethod
    @typing.override
    def _is_val_type(cls, val) -> TypeIs[Callable[..., Any]]:
        return callable(val)


class _TypeRegistry(_TypeCheckDict[str, Callable]):
    @classmethod
    @typing.override
    def _is_key_type(cls, key) -> TypeIs[str]:
        return isinstance(key, str)

    @classmethod
    @typing.override
    def _is_val_type(cls, val) -> TypeIs[Callable[..., Any]]:
        return callable(val)


class _GlobalRegistry(_TypeCheckDict[type, _TypeRegistry]):
    def __rich__(self):
        raise NotImplementedError

    @typing.override
    def __getitem__(self, key: type):
        # Make it behave like a default dict.
        if key not in self:
            self[key] = _TypeRegistry()

        # This also does type check!
        return super().__getitem__(key)

    @classmethod
    @typing.override
    def _is_key_type(cls, key) -> TypeIs[type]:
        return isinstance(key, type)

    @classmethod
    @typing.override
    def _is_val_type(cls, val) -> TypeIs[_TypeRegistry]:
        return isinstance(val, _TypeRegistry)


_GLOBAL_REGISTRY = _GlobalRegistry()
"The type based registry."


_REGISTRY_2D = _Registry2D()

type _UnaryFunc[T] = Callable[[T], T]
type _BinaryFunc[T] = Callable[[T, T], T]
type _AnyFunc[T] = Callable[..., T]


@typing.no_type_check
def register_1[T](typ: type, op: str) -> Callable[[_UnaryFunc[T]], _UnaryFunc[T]]:
    "Register a unary function."

    return register(RegKey(typ=typ, op=op))


@typing.no_type_check
def register_2[T](typ: type, op: str) -> Callable[[_BinaryFunc[T]], _BinaryFunc[T]]:
    "Register a binary function."

    return register(RegKey(typ=typ, op=op))


def register[T](key: RegKey):
    """
    Register a function. If argc is given, the signature should match.
    """

    def regsiterer(func: Callable[..., T]):
        _GLOBAL_REGISTRY[key.typ][key.op] = func
        return func

    return regsiterer


def find_1(op: str, typ: type):
    return find(op=op, typ=typ)


def find_2(op: str, typ: type):
    return find(op=op, typ=typ)


def find(op: str, typ: type) -> Callable:
    """
    Find the given op, typ pair in the registry.

    If ``argc`` is given, the result must satisfy the argument count,
    or else a ``ValueError`` is raised. By default this check is off.
    """

    return _GLOBAL_REGISTRY[typ][op]


def types():
    return frozenset(_GLOBAL_REGISTRY.keys())


def operators():
    result: set[str] = set()
    for op in _GLOBAL_REGISTRY.values():
        result.update(op.keys())
    return result
