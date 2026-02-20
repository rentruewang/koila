# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest

from aioway import attrs
from aioway._kernels import Array, BroadCastSameKernel


@dcls.dataclass(frozen=True)
class BcastCase:
    left: Array
    right: Array
    output: Array

    def __iter__(self):
        yield self.left
        yield self.right
        yield self.output


@pytest.mark.parametrize(
    "case",
    [
        BcastCase(
            left=Array(shape=[1, 2, 4], dtype=attrs.dtype("int64"), cost=0),
            right=Array(shape=[1, 2, 4], dtype=attrs.dtype("int64"), cost=0),
            output=Array(shape=[1, 2, 4], dtype=attrs.dtype("int64"), cost=8 * 64),
        ),
    ],
)
def test_bcast_same(case):
    left, right, out = case
    kernel = BroadCastSameKernel(left, right)
    assert kernel.compute() == out
