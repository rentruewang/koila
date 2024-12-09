# Copyright (c) RenChu Wang - All Rights Reserved

from aioway.schemas import ArrayDtype, BoolDtype, DataTypeEnum, FloatDtype, IntDtype


def test_builders():
    assert DataTypeEnum.INT() == IntDtype()
    assert DataTypeEnum.FLOAT() == FloatDtype()
    assert DataTypeEnum.BOOL() == BoolDtype()
    assert DataTypeEnum.ARRAY() == ArrayDtype()

    assert DataTypeEnum.INT(64) == IntDtype(64)
    assert DataTypeEnum.FLOAT(32) == FloatDtype(32)
    assert DataTypeEnum.ARRAY((1, 2, 3)) == ArrayDtype((1, 2, 3))
