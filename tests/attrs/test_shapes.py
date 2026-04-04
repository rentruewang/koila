# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest

from aioway import meta


def _shapes():
    yield meta.Shape.parse(3, 5, 7)
    yield meta.Shape.parse([3, 5, 7])


@pytest.fixture(params=_shapes())
def shape(request: pytest.FixtureRequest) -> meta.Shape:
    return request.param


def test_shape_getitem(shape: meta.Shape):
    assert shape[0] == 3
    assert shape[1] == 5
    assert shape[2] == 7


def test_shape_size(shape: meta.Shape):
    assert shape.size == 3 * 5 * 7


def test_shape_ndim(shape: meta.Shape):
    assert shape.ndim == 3


def test_shape_no_fail(shape: meta.Shape):
    "Ensure that shapes do not crash when the other type is not recognized."

    assert shape != object()


def test_shape_ndarray(shape: meta.Shape):
    assert shape == np.array([3, 5, 7])
    assert shape != np.array([[3, 5, 7]])
