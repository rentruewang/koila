# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest
import tensordict as td
import torch

from aioway import chunks, tdicts, tensors
from aioway.tdicts import _validation


@pytest.fixture
def schema() -> tdicts.AttrSet:
    return tdicts.AttrSet.from_values(
        a=tensors.attr(
            {
                "device": "cpu",
                "dtype": "int32",
                "max_shape": [-1, 2, 3],
            },
        ),
        b=tensors.attr(
            {
                "device": "cpu",
                "dtype": "float32",
                "max_shape": [-1, 6],
            },
        ),
    )


@pytest.fixture
def valid_data() -> td.TensorDict:
    result = td.TensorDict(
        {
            "a": torch.randn(11, 2, 3).to(torch.int32),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    )
    result.auto_batch_size_()
    return result


def _invalid_data():
    # Invalid shape
    yield td.TensorDict(
        {
            "a": torch.randn(11, 2, 3, 4).to(torch.int32),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    ).auto_batch_size_()

    # Invalid dtype
    yield td.TensorDict(
        {
            "a": torch.randn(11, 2, 3).to(torch.int64),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    ).auto_batch_size_()


@pytest.fixture(params=_invalid_data())
def invalid_data(request: pytest.FixtureRequest) -> td.TensorDict:
    return request.param


def test_attrset_getitem(schema: tdicts.AttrSet):
    assert isinstance(schema["a"], tensors.Attr)
    assert isinstance(schema[["a", "b"]], tdicts.AttrSet)
    assert schema == schema[["a", "b"]]
    assert isinstance(schema[[-1, 2, 3]], tdicts.AttrSet)
    assert isinstance(schema[np.array([-1, 2, 3])], tdicts.AttrSet)


def test_validation_ok(schema: tdicts.AttrSet, valid_data: td.TensorDict) -> None:
    _validation.validate_schema(schema, valid_data)


def test_construction_of_attrset(valid_data: td.TensorDict):
    parsed = tdicts.tdict(valid_data)
    assert parsed.attrs == tdicts.attr_set(
        {
            "a": tensors.Attr.parse(device="cpu", max_shape=[11, 2, 3], dtype="int32"),
            "b": tensors.Attr.parse(device="cpu", max_shape=[11, 6], dtype="float32"),
        }
    )


def test_validation_fail(schema: tdicts.AttrSet, invalid_data: td.TensorDict):
    with pytest.raises(RuntimeError):
        _validation.validate_schema(schema, invalid_data)


@pytest.fixture
def block(schema: tdicts.AttrSet, valid_data: td.TensorDict) -> chunks.Chunk:
    return chunks.Chunk.from_data_schema(data=valid_data, schema=schema)


def test_block_init(block: chunks.Chunk):
    _ = block
