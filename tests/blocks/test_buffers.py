# Copyright (c) RenChu Wang - All Rights Reserved

import pytest
import torch

from aioway.blocks import Buffer
from aioway.schemas import DataTypeEnum


@pytest.fixture
def left_buf():
    return Buffer(torch.randn(97, 101).float())


@pytest.fixture
def right_buf():
    return Buffer(torch.randn(97, 101).float())


def test_dtype(left_buf):
    assert left_buf.dtype.shape == (97, 101)
    assert left_buf.dtype.dtype == DataTypeEnum.FLOAT(32)


def test_arithmetic(left_buf, right_buf):
    assert left_buf + right_buf == Buffer(left_buf.data + right_buf.data)
    assert left_buf - right_buf == Buffer(left_buf.data - right_buf.data)
    assert left_buf * right_buf == Buffer(left_buf.data * right_buf.data)
    assert left_buf / right_buf == Buffer(left_buf.data / right_buf.data)
    assert left_buf // right_buf == Buffer(left_buf.data // right_buf.data)
    assert left_buf % right_buf == Buffer(left_buf.data % right_buf.data)
    assert left_buf**right_buf == Buffer(left_buf.data**right_buf.data)


def test_comparison(left_buf, right_buf):
    assert (left_buf == right_buf) == Buffer(left_buf.data == right_buf.data)
    assert (left_buf != right_buf) == Buffer(left_buf.data != right_buf.data)
    assert (left_buf > right_buf) == Buffer(left_buf.data > right_buf.data)
    assert (left_buf < right_buf) == Buffer(left_buf.data < right_buf.data)
    assert (left_buf >= right_buf) == Buffer(left_buf.data >= right_buf.data)
    assert (left_buf <= right_buf) == Buffer(left_buf.data <= right_buf.data)


def test_incompatible(left_buf, right_buf):
    with pytest.raises(RuntimeError):
        left_buf + Buffer(right_buf.data.T)
