# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
import typing
from collections import UserDict
from collections.abc import Callable
from typing import Any, Literal, NamedTuple

from rich.table import Table

__all__ = ["registry", "register_1", "register_2", "find", "find_1", "find_2"]

type _Argc = Literal[1, 2, -1]


class RegKey(NamedTuple):
    "The registry key in tuple form."

    op: str
    """
    The operator's name. Corresponds to the row of the specialization table.
    """

    typ: type
    """
    The type of the operator. Corresponds to the columns of the specialization table.
    """

    def signature_or_missing(self) -> str:
        return render_key_signature_or_missing(self)


class _Registry(UserDict[RegKey, Callable]):
    def __rich__(self):
        return _rich_table()

    def __getitem__(self, key: RegKey):
        self.__validate_key(key)
        return super().__getitem__(key)

    def __setitem__(self, key: RegKey, item: Callable[..., Any]) -> None:
        self.__validate_key(key)
        self.__validate_value(item)
        return super().__setitem__(key, item)

    def __delitem__(self, key: RegKey) -> None:
        self.__validate_key(key)
        super().__delitem__(key)

    def __validate_key(self, key):
        if not isinstance(key, RegKey):
            raise TypeError(key)

    def __validate_value(self, value):
        if not callable(value):
            raise ValueError(value)


_REGISTRY = _Registry()
"The main registry. Might change in the future."

_OP_SIGNATURE: dict[str, _Argc] = {}
"The signature metadata for the operators."


def registry():
    return _REGISTRY


type _UnaryFunc[T] = Callable[[T], T]
type _BinaryFunc[T] = Callable[[T, T], T]
type _AnyFunc[T] = Callable[..., T]


@typing.no_type_check
def register_1[T](op: str, typ: type) -> Callable[[_UnaryFunc[T]], _UnaryFunc[T]]:
    "Register a unary function."

    return register(RegKey(op, typ), 1)


@typing.no_type_check
def register_2[T](op: str, typ: type) -> Callable[[_BinaryFunc[T]], _BinaryFunc[T]]:
    "Register a binary function."

    return register(RegKey(op, typ), 2)


def register[T](key: RegKey, argc: _Argc = -1):
    """
    Register a function. If argc is given, the signature should match.
    """

    def regsiterer(func: Callable[..., T]):
        _validate_signature(key.op, func, argc=argc)
        _add_to_mapping(key, func)
        return func

    return regsiterer


def find_1(op: str, typ: type):
    return find(op=op, typ=typ, argc=1)


def find_2(op: str, typ: type):
    return find(op=op, typ=typ, argc=2)


def find(op: str, typ: type, argc: _Argc = -1) -> Callable:
    """
    Find the given op, typ pair in the registry.

    If ``argc`` is given, the result must satisfy the argument count,
    or else a ``ValueError`` is raised. By default this check is off.
    """

    _check_argc_compat(argc, _OP_SIGNATURE[op])

    return _REGISTRY[RegKey(op=op, typ=typ)]


def _add_to_mapping(key: RegKey, func: Callable):
    if key in _REGISTRY:
        raise KeyError(f"Already registered: {key=}, value={_REGISTRY[key]}.")

    _REGISTRY[key] = func


def _validate_signature(op: str, func: Callable, argc: _Argc) -> None:
    """
    If ``op`` is previously given, ``argc`` would be check against its previous value.
    Then, validate the ``func`` with the ``argc``.

    Raises:
        ValueError: If the ``argc`` has a mismatch.
    """

    if op not in _OP_SIGNATURE:
        _OP_SIGNATURE[op] = argc

    # Check if the given argc is compatible with previous.
    _check_argc_compat(_OP_SIGNATURE[op], argc)

    # Check if the signature is compatible with the given.
    _check_argc_compat(argc, len(inspect.signature(func).parameters))


def _check_argc_compat(argc: _Argc, argc_to_check: int, /) -> None:
    "Check if the ``argc_to_check`` is compatible with the ``argc``."

    match argc:
        case 1 | 2:
            # The signature must match.
            if argc_to_check != argc:
                raise ValueError(f"Expected {argc=}, got argc={argc_to_check}.")
        case -1:
            # Any args is ok.
            return


def types():
    return {key.typ for key in _REGISTRY.keys()}


def operators():
    return {key.op for key in _REGISTRY.keys()}


def _rich_table():
    types_set = types()

    table = Table("op", *(typ.__name__ for typ in types_set))

    for op in operators():
        _add_table_types_row(types_set, table, op)

    return table


def _add_table_types_row(types_set: set[type], table: Table, op: str):
    variants = [RegKey(op=op, typ=typ).signature_or_missing() for typ in types_set]
    table.add_row(op, *variants)


def render_key_signature_or_missing(key: RegKey) -> str:
    "If key is missing in registry, render '-', or else render the signature."
    if key in _REGISTRY:
        type_name = key.typ.__name__
        args_repr = _args_body_str(_OP_SIGNATURE[key.op], type_name)
        return f"{key.op}({args_repr})"
    else:
        return "-"


def _args_body_str(argc: _Argc, typ: str):
    match argc:
        case 1:
            return typ
        case 2:
            return typ + "," + typ
        case -1:
            return typ + "..."
