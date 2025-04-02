# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import functools
from pathlib import Path
from typing import Protocol, Self

import lark
from lark import Lark, Transformer

__all__ = ["SchemaTypeParser", "SchemaTypeTransformer"]


class _Stringer(Protocol):
    """
    ``Stringer`` is something that can be converted to a string.
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
    def default(
        cls,
        grammar: str | Path = Path(__file__).parent / "datatypes.lark",
        start: str = "start",
    ) -> Self:
        text = Path(grammar).read_text()
        parser = Lark(grammar=text, start=start)
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
