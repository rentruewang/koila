# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
from pytest import FixtureRequest

from aioway.attrs import Shape


def _shapes():
    yield Shape(3, 5, 7)
    yield Shape.wrap([3, 5, 7])


@pytest.fixture(params=_shapes())
def shape(request: FixtureRequest):
    return request.param


def test_shape_getitem(shape: Shape):
    assert shape[0] == 3
    assert shape[1] == 5
    assert shape[2] == 7


def test_shape_size(shape: Shape):
    assert shape.size == 3 * 5 * 7


def test_shape_ndim(shape: Shape):
    assert shape.ndim == 3
