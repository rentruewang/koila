# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import functools
from typing import Protocol, Self

import lark
from lark import Lark, Transformer

__all__ = ["SchemaTypeParser", "SchemaTypeTransformer"]

SCHEMA_TYPE_GRAMMAR = r"""
?start: bigint
     | bit
     | blob
     | boolean
     | float64 | float32 | float16
     | int64 | int32 | int16 | int8
     | interval
     | json
     | date
     | string
     | decimal


bigint: "BIGINT"i | "INT8"i | "LONG"i | "HUGEINT"i
bit: "BIT"i | "BITSTRING"i
blob: "BLOB"i | "BYTEA"i | "BINARY"i | "VARBINARY"i
boolean: "BOOLEAN"i | "BOOL"i | "LOGICAL"i
float64: "DOUBLE"i | "FLOAT64"i
float32: "FLOAT"i | "FLOAT32"i
float16: "HALF"i | "FLOAT16"i
int64: "INTEGER"i | "INT"i | "INT64"i
int32: "INT32"i
int16: "INT16"i
int8: "INT8"i | "CHAR"i
interval: "INTERVAL"i
json: "JSON"i

date: "DATE" | "DATETIME" | "TIMESTAMP"

string: "VARCHAR"i | "STR"i | "BPCHAR"i | "TEXT"i | "STRING"i

precision: INT
scale: INT

decimal: "DECIMAL" "(" precision "," scale ")"

%import common.WS
%ignore WS

%import common.INT
%import common.SIGNED_NUMBER    -> NUMBER
"""


class _Stringer(Protocol):
    """
    ``Stringer`` is something that can be converted to a string.

    This is used as a common interface to accept both
    objects that define ``__str__``, and ``str`` objects themselves.
    """

    def __str__(self) -> str: ...


@dcls.dataclass(frozen=True)
class SchemaTypeParser:
    parser: Lark
    """
    The lark parser to use.
    """

    def __call__(self, text: str) -> _Stringer:
        parsed = self.parser.parse(text)
        return self._transformer.transform(parsed)

    @functools.cached_property
    def _transformer(self) -> Transformer:
        return SchemaTypeTransformer()

    @classmethod
    def default(cls, grammar: str = SCHEMA_TYPE_GRAMMAR) -> Self:
        parser = Lark(grammar=grammar)
        return cls(parser=parser)


@dcls.dataclass(frozen=True, repr=False)
class Decimal:
    precision: int
    scale: int

    def __repr__(self) -> str:
        return f"DECIMAL({self.precision}, {self.scale})"


@dcls.dataclass(frozen=True, repr=False)
class Float:
    precision: int = 32

    def __repr__(self):
        return f"FLOAT{self.precision}"


@dcls.dataclass(frozen=True, repr=False)
class Int:
    precision: int = 64

    def __repr__(self):
        return f"INT{self.precision}"


def _type_no_arg_fn(name: _Stringer):
    """
    A simple wrapper to generate a member method, returning a specific value.
    This is useful as ``lark``'s rules to generate some values,
    ones that are only dependent on what rules are called but not the value of the rules.
    """

    def func(_) -> _Stringer:
        return name

    return func


@lark.v_args(inline=True)
class SchemaTypeTransformer(Transformer):
    bigint = _type_no_arg_fn("BIGINT")
    bit = _type_no_arg_fn("BIT")
    blob = _type_no_arg_fn("BLOB")
    boolean = _type_no_arg_fn("BOOL")
    string = _type_no_arg_fn("STRING")
    json = _type_no_arg_fn("JSON")
    float64 = _type_no_arg_fn(Float(64))
    float32 = _type_no_arg_fn(Float(32))
    float16 = _type_no_arg_fn(Float(16))
    int64 = _type_no_arg_fn(Int(64))
    int32 = _type_no_arg_fn(Int(32))
    int16 = _type_no_arg_fn(Int(16))
    int8 = _type_no_arg_fn(Int(8))
    interval = _type_no_arg_fn("INTERVAL")
    date = _type_no_arg_fn("DATE")

    precision = scale = int
    decimal = Decimal
