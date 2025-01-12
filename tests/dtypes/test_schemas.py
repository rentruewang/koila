# Copyright (c) RenChu Wang - All Rights Reserved

import pytest

from aioway.schemas import DataType, DataTypeEnum, Schema


@pytest.fixture
def schema_dict() -> dict[str, DataType]:
    return {"hello": DataTypeEnum.INT(), "world": DataTypeEnum.FLOAT()}


@pytest.fixture
def schema(schema_dict) -> Schema:
    return Schema.mapping(schema_dict)


def test_init(schema_dict, schema):
    assert Schema.mapping(schema_dict) == Schema.iterable(schema.sorted())
    assert Schema.mapping(schema_dict) == Schema.tuples(list(schema_dict.items()))


def test_eq(schema_dict, schema):
    assert schema == schema_dict
    assert schema == Schema.mapping(schema_dict)


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
    assert [col.name for col in schema.sorted()] == sorted(schema.names)


def test_dtypes(schema_dict, schema):
    assert set(schema.dtypes) == set(schema_dict.values())
    assert {col.dtype for col in schema.sorted()} == set(schema.dtypes)
