# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import pytest

from aioway.schemas import DataType, DataTypeEnum, TableSchema


@pytest.fixture
def schema_dict() -> dict[str, DataType]:
    return {"hello": DataTypeEnum.INT(), "world": DataTypeEnum.FLOAT()}


@pytest.fixture
def schema(schema_dict) -> TableSchema:
    return TableSchema.mapping(schema_dict)


def test_init(schema_dict, schema):
    assert TableSchema.mapping(schema_dict) == TableSchema.iterable(schema.ordered())
    assert TableSchema.mapping(schema_dict) == TableSchema.tuples(
        list(schema_dict.items())
    )


def test_eq(schema_dict, schema):
    assert schema == schema_dict
    assert schema == TableSchema.mapping(schema_dict)


def test_string(schema):
    assert str(schema) == r"{hello: int64, world: float32}"


def test_mapping(schema):
    assert len(schema) == 2
    assert schema["hello"] == DataTypeEnum.INT()
    assert schema["world"] == DataTypeEnum.FLOAT()
    assert "hello" in schema
    assert "hi" not in schema


def test_names(schema):
    assert sorted(schema.names) == ["hello", "world"]
    assert [col.name for col in schema.ordered()] == sorted(schema.names)


def test_dtypes(schema_dict, schema):
    assert set(schema.dtypes) == set(schema_dict.values())
    assert {col.dtype for col in schema.ordered()} == set(schema.dtypes)


def test_null():
    assert TableSchema.null() == TableSchema.mapping({})
    assert TableSchema.null() == TableSchema.iterable([])
