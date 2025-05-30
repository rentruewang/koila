# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.schemas import SchemaTypeParser


def example_types_and_transformed():
    return [
        ["BIGINT", "BIGINT"],
        ["BIT", "BIT"],
        ["BLOB", "BLOB"],
        ["BOOLEAN", "BOOL"],
        ["BOOL", "BOOL"],
        ["LOGICAL", "BOOL"],
        ["DATE", "DATE"],
        ["DECIMAL(32, 5)", "DECIMAL(32, 5)"],
        ["FLOAT", "FLOAT32"],
        ["FLOAT32", "FLOAT32"],
        ["DOUBLE", "FLOAT64"],
        ["FLOAT64", "FLOAT64"],
        ["INTEGER", "INT64"],
        ["INT", "INT64"],
        ["CHAR", "INT8"],
    ]


@pytest.fixture(scope="module", params=example_types_and_transformed())
def schema(request):
    return request.param


@pytest.fixture(scope="module")
def parser():
    return SchemaTypeParser.default()


def test_schema_parser(schema, parser):
    input_name, true_name = schema
    parsed = parser(input_name)
    assert str(parsed) == true_name
