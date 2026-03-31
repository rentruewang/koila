# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
import pytest

from aioway import tensors


def _shapes():
    yield tensors.Shape.parse(3, 5, 7)
    yield tensors.Shape.parse([3, 5, 7])


@pytest.fixture(params=_shapes())
def shape(request: pytest.FixtureRequest) -> tensors.Shape:
    return request.param


def test_shape_getitem(shape: tensors.Shape):
    assert shape[0] == 3
    assert shape[1] == 5
    assert shape[2] == 7


def test_shape_size(shape: tensors.Shape):
    assert shape.size == 3 * 5 * 7


def test_shape_ndim(shape: tensors.Shape):
    assert shape.ndim == 3


def test_shape_no_fail(shape: tensors.Shape):
    "Ensure that shapes do not crash when the other type is not recognized."

    assert shape != object()


def test_shape_ndarray(shape: tensors.Shape):
    assert shape == np.array([3, 5, 7])
    assert shape != np.array([[3, 5, 7]])
