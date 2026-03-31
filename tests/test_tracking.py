# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import tensors
from aioway._tracking import logging


@pytest.fixture(scope="module", autouse=True)
def enable_rich():
    with logging.enable_rich_log("DEBUG"):
        yield


@pytest.fixture
def lhs():
    return tensors.Attr.parse(dtype="int8", device="cpu", shape=[1, 2, 3])


@pytest.fixture
def rhs():
    return tensors.Attr.parse(dtype="float16", device="cpu", shape=[1, 1, 3])


def test_attr_binary(lhs: tensors.Attr, rhs: tensors.Attr):
    result = lhs.term + rhs
    assert isinstance(result.unpack(), tensors.Attr)


def test_attr_unary(lhs: tensors.Attr):
    result = ~lhs.term
    assert isinstance(result.unpack(), tensors.Attr)
