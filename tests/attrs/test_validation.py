# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import torch
from tensordict import TensorDict

from aioway import attrs
from aioway.attrs import AttrSet, _validation
from aioway.batches import Chunk


@pytest.fixture
def schema():
    return AttrSet.from_values(
        a=attrs.attr(
            device="cpu",
            dtype="int32",
            shape=[2, 3],
        ),
        b=attrs.attr(
            device="cpu",
            dtype="float32",
            shape=[6],
        ),
    )


@pytest.fixture
def valid_data():
    result = TensorDict(
        {
            "a": torch.randn(11, 2, 3).to(torch.int32),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    )
    result.auto_batch_size_()
    return result


def _invalid_data():
    # Invalid shape
    yield TensorDict(
        {
            "a": torch.randn(11, 2, 3, 4).to(torch.int32),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    ).auto_batch_size_()

    # Invalid dtype
    yield TensorDict(
        {
            "a": torch.randn(11, 2, 3).to(torch.int64),
            "b": torch.randn(11, 6).to(torch.float32),
        }
    ).auto_batch_size_()


@pytest.fixture(params=_invalid_data())
def invalid_data(request) -> TensorDict:
    return request.param


def test_validation_ok(schema, valid_data):
    _validation.validate_schema(schema, valid_data)


def test_validation_fail(schema, invalid_data):
    with pytest.raises(RuntimeError):
        _validation.validate_schema(schema, invalid_data)


@pytest.fixture
def block(schema, valid_data):
    return Chunk.from_data_schema(data=valid_data, schema=schema)


def test_block_init(block):
    _ = block
