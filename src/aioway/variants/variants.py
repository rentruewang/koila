# Copyright (c) AIoWay Authors - All Rights Reserved

import inspect
from collections import defaultdict as DefaultDict
from collections.abc import Callable
from typing import Any, NamedTuple

from rich.table import Table

__all__ = ["VariantRegistry", "UNARY_UFUNCS", "BINARY_UFUNCS"]


class _OpType(NamedTuple):
    op: str
    typ: type


class VariantRegistry[T: Callable]:
    """
    ``VariantRegistry`` is a 2D table of op (str) vs class (type).
    """

    def __init__(self, num_args: int):
        if num_args < 0:
            raise ValueError(f"Number of arguments should be >= 0, got {num_args=}")

        self._num_args = num_args

        # The registry mappings to use.
        self._ops_mapping: dict[str, set[T]] = DefaultDict(set)
        self._types_mapping: dict[type, set[T]] = DefaultDict(set)
        self._ops_types_mapping: dict[_OpType, T] = {}

    def __rich__(self):
        types = list(self.types())

        table = Table("op", *(typ.__name__ for typ in types))

        for op in self.operators():
            items = [self._format_variant(op, typ) for typ in types]
            table.add_row(op, *items)

        return table

    def _register_func(self, op: str, typ: type, func: T) -> None:
        signature = inspect.signature(func)
        if len(signature.parameters) != self._num_args:
            func_name = f"{func.__module__}.{func.__qualname__}"
            raise ValueError(
                f"Expect function '{func_name}' argument count to be {self._num_args}, "
                f"but got {len(signature.parameters)} arguments."
            )

        self._ops_types_mapping[_OpType(op=op, typ=typ)] = func
        self._ops_mapping[op].add(func)
        self._types_mapping[typ].add(func)

    def register(self, op: str, typ: type):
        def register_func(func: T) -> T:
            self._register_func(op=op, typ=typ, func=func)
            return func

        return register_func

    def find(self, op: str, typ: type) -> T:
        return self._ops_types_mapping[_OpType(op=op, typ=typ)]

    def operators(self) -> list[str]:
        return list(self._ops_mapping.keys())

    def types(self) -> list[type]:
        return list(self._types_mapping.keys())

    def _format_variant(self, op: str, typ: type):
        key = _OpType(op, typ)

        if key in self._ops_types_mapping:
            return f"{op}<{typ.__name__}>"
        else:
            return "-"


UNARY_UFUNCS = VariantRegistry[Callable[[Any], Any]](num_args=1)
BINARY_UFUNCS = VariantRegistry[Callable[[Any, Any], Any]](num_args=2)
