# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from aioway.schemas import DataTypeEnum


def test_dtype_reprs():
    assert str(DataTypeEnum.INT()) == "int64"
    assert str(DataTypeEnum.INT[64]()) == "int64"
    assert str(DataTypeEnum.INT[32]()) == "int32"
    assert str(DataTypeEnum.INT[16]()) == "int16"

    assert str(DataTypeEnum.FLOAT()) == "float32"
    assert str(DataTypeEnum.FLOAT[64]()) == "float64"
    assert str(DataTypeEnum.FLOAT[32]()) == "float32"
    assert str(DataTypeEnum.FLOAT[16]()) == "float16"

    assert str(DataTypeEnum.BOOL()) == "bool"

    assert str(DataTypeEnum.ARRAY()) == "array(shape=(), dtype=float32)"
    assert str(DataTypeEnum.ARRAY[1, 2]()) == "array(shape=(1, 2), dtype=float32)"


def test_dtype_eq():
    assert DataTypeEnum.INT() == "int64"
    assert DataTypeEnum.INT() == DataTypeEnum.INT[64]()
    assert DataTypeEnum.INT[64]() == "int64"
    assert DataTypeEnum.INT[32]() == "int32"
    assert DataTypeEnum.INT[16]() == "int16"

    assert DataTypeEnum.FLOAT() == "float32"
    assert DataTypeEnum.FLOAT() == DataTypeEnum.FLOAT[32]()
    assert DataTypeEnum.FLOAT[64]() == "float64"
    assert DataTypeEnum.FLOAT[32]() == "float32"
    assert DataTypeEnum.FLOAT[16]() == "float16"

    assert DataTypeEnum.BOOL() == "bool"

    assert DataTypeEnum.ARRAY() == "array(shape=(), dtype=float32)"
    assert DataTypeEnum.ARRAY[1, 2]() == "array(shape=(1, 2), dtype=float32)"
