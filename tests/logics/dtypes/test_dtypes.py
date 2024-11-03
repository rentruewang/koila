# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from aioway.logics import (
    ArrayDtype,
    BoolDtype,
    DataType,
    DtypeFactory,
    FloatDtype,
    IntDtype,
    StrDtype,
)


class DtypeStringer(DataType.Visitor[str]):
    def boolean(self, dtype: BoolDtype) -> str:
        _ = dtype
        return "bool"

    def integer(self, dtype: IntDtype) -> str:
        return f"int{dtype.precision}"

    def floating(self, dtype: FloatDtype) -> str:
        return f"float{dtype.precision}"

    def array(self, dtype: ArrayDtype) -> str:
        return f"array:{dtype.shape}"

    def string(self, dtype: StrDtype) -> str:
        return f"str({dtype.length})"


def test_dtype_visitors():
    stringer = DtypeStringer()

    assert stringer(DtypeFactory.INT()) == "int64"
    assert stringer(DtypeFactory.INT[64]()) == "int64"
    assert stringer(DtypeFactory.INT[32]()) == "int32"
    assert stringer(DtypeFactory.INT[16]()) == "int16"

    assert stringer(DtypeFactory.FLOAT()) == "float32"
    assert stringer(DtypeFactory.FLOAT[64]()) == "float64"
    assert stringer(DtypeFactory.FLOAT[32]()) == "float32"
    assert stringer(DtypeFactory.FLOAT[16]()) == "float16"

    assert stringer(DtypeFactory.BOOL()) == "bool"

    assert stringer(DtypeFactory.STR()) == "str(None)"
    assert stringer(DtypeFactory.STR[1234]()) == "str(1234)"

    assert stringer(DtypeFactory.ARRAY()) == "array:None"
    assert stringer(DtypeFactory.ARRAY[1, 2, 3]()) == "array:(1, 2, 3)"
