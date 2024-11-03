# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from aioway.logics import (
    ArrayDtype,
    BoolDtype,
    DtypeFactory,
    FloatDtype,
    IntDtype,
    StrDtype,
)


def test_builders():
    assert DtypeFactory.INT() == IntDtype()
    assert DtypeFactory.FLOAT() == FloatDtype()
    assert DtypeFactory.BOOL() == BoolDtype()
    assert DtypeFactory.STR() == StrDtype()
    assert DtypeFactory.ARRAY() == ArrayDtype()

    assert DtypeFactory.INT[64]() == IntDtype(64)
    assert DtypeFactory.FLOAT[32]() == FloatDtype(32)
    assert DtypeFactory.ARRAY[1, 2, 3]() == ArrayDtype(shape=(1, 2, 3))
