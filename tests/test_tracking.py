# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._tracking import logging
from aioway.attrs import Attr


@pytest.fixture(scope="module", autouse=True)
def enable_rich():
    with logging.enable_rich_log("DEBUG"):
        yield


@pytest.fixture
def lhs():
    return Attr.parse(dtype="int8", device="cpu", shape=[1, 2, 3])


@pytest.fixture
def rhs():
    return Attr.parse(dtype="float16", device="cpu", shape=[1, 1, 3])


def test_attr_binary(lhs: Attr, rhs: Attr):
    result = lhs.term + rhs
    assert isinstance(result.unpack(), Attr)


def test_attr_unary(lhs: Attr):
    result = ~lhs.term
    assert isinstance(result.unpack(), Attr)
