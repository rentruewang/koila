# Copyright (c) AIoWay Authors - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.attrs import Shape


def sample_shapes() -> list[list[int]]:
    with open(Path(__file__).parent / "shapes.json") as f:
        return json.load(f)


@pytest.fixture(params=sample_shapes(), scope="module")
def shape(request) -> list[int]:
    return request.param


def test_shapes_shape(shape):
    assert isinstance(shape, list)
    assert isinstance(Shape.from_iterable(shape), Shape)


def test_shapes_eq(shape):
    assert len(Shape.from_iterable(shape)) == len(shape)
    assert Shape.from_iterable(shape) == list(shape)
    assert Shape.from_iterable(shape) == tuple(shape)
