# Copyright (c) RenChu Wang - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.datatypes import Shape


def sample_shapes() -> list[list[int]]:
    with open(Path(__file__).parent / "shapes.json") as f:
        return json.load(f)


@pytest.fixture(params=sample_shapes(), scope="module")
def shape(request) -> list[int]:
    return request.param


def test_shapes_shape(shape):
    assert isinstance(shape, list)
    assert isinstance(Shape.from_seq(shape), Shape)


def test_shapes_eq(shape):
    assert len(Shape.from_seq(shape)) == len(shape)
    assert Shape.from_seq(shape) == list(shape)
    assert Shape.from_seq(shape) == tuple(shape)
