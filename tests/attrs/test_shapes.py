# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway import attrs
from aioway.attrs import Shape


def _shapes():
    yield attrs.shape(3, 5, 7)
    yield attrs.shape([3, 5, 7])


@pytest.fixture(params=_shapes())
def shape(request) -> Shape:
    return request.param


def test_shape_getitem(shape):
    assert shape[0] == 3
    assert shape[1] == 5
    assert shape[2] == 7


def test_shape_size(shape):
    assert shape.size == 3 * 5 * 7


def test_shape_ndim(shape):
    assert shape.ndim == 3
