# Copyright (c) AIoWay Authors - All Rights Reserved

import typing
from collections import abc as cabc

__all__ = [
    "is_list_of",
    "is_tuple_of",
    "is_seq_of",
    "is_dict_of_str_to",
]


@typing.no_type_check
def _seq_check[T](seq: type, typ: type[T]):
    if not issubclass(seq, cabc.Sequence):
        raise TypeError(
            f"The given seq: `{seq}` should be subclass of `cabc.Sequence`."
        )

    if not isinstance(typ, type):
        raise TypeError(f"The given typ: `{typ}` should be a type.")

    def check(obj) -> typing.TypeGuard[typing.Any]:
        return isinstance(obj, seq) and all(isinstance(i, typ) for i in obj)

    return check


@typing.no_type_check
def _mapping_check[K, V](mapping: type, key: type[K], val: type[V]):
    if not issubclass(mapping, cabc.Mapping):
        raise TypeError(
            f"The given mapping: `{mapping}` should be subclass of `cabc.Mapping`."
        )

    if not isinstance(key, type):
        raise TypeError(f"The given key: `{key}` should be a type.")

    if not isinstance(val, type):
        raise TypeError(f"The given val: `{val}` should be a type.")

    def check(obj) -> typing.TypeGuard[typing.Any]:
        return isinstance(obj, mapping) and all(
            isinstance(k, key) and isinstance(v, val) for k, v in obj.items()
        )

    return check


def is_seq_of[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[cabc.Sequence[T]]]:
    return _seq_check(cabc.Sequence, typ)


def is_list_of[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[list[T]]]:
    return _seq_check(list, typ)


def is_tuple_of[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[tuple[T, ...]]]:
    return _seq_check(tuple, typ)


def is_dict_of_str_to[T](
    typ: type[T], /
) -> cabc.Callable[[typing.Any], typing.TypeGuard[dict[str, T]]]:
    return _mapping_check(dict, str, typ)
