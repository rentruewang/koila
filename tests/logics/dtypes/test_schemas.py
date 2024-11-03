# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from aioway.logics import DataType, DtypeFactory, Schema


def _schema_dict() -> dict[str, DataType]:
    return {"hello": DtypeFactory.INT(), "world": DtypeFactory.FLOAT()}


def _schema() -> Schema:
    return Schema.mapping(_schema_dict())


def test_init():
    schema_dict = _schema_dict()
    schema = _schema()

    assert Schema.mapping(schema_dict) == Schema.iterable(schema.sorted())
    assert Schema.mapping(schema_dict) == Schema.tuples(list(schema_dict.items()))


def test_eq():
    schema_dict = _schema_dict()
    schema = _schema()

    assert schema == schema_dict
    assert schema == Schema.mapping(schema_dict)


def test_string():
    schema = _schema()
    assert (
        str(schema)
        == r"{hello: IntDtype(precision=64), world: FloatDtype(precision=32)}"
    )


def test_mapping():
    schema = _schema()
    assert len(schema) == 2
    assert schema["hello"] == DtypeFactory.INT()
    assert schema["world"] == DtypeFactory.FLOAT()
    assert "hello" in schema
    assert "hi" not in schema


def test_names():
    schema = _schema()
    assert sorted(schema.names) == ["hello", "world"]
    assert [col.name for col in schema.sorted()] == sorted(schema.names)


def test_dtypes():
    schema = _schema()
    assert set(schema.dtypes) == set(_schema_dict().values())
    assert {col.dtype for col in schema.sorted()} == set(schema.dtypes)


def test_null():
    assert Schema.null() == Schema.mapping({})
    assert Schema.null() == Schema.iterable([])
