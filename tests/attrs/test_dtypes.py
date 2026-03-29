# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
from typing import Any

import numpy as np
import pytest
import torch
from pytest import FixtureRequest

from aioway.tensors import DType


@dcls.dataclass(frozen=True)
class _CaseChecker:
    original: Any
    family: str
    bits: int

    @property
    def dtype(self):
        return DType.parse(self.original)

    def check(self):
        assert isinstance(self.dtype, DType)
        assert self.dtype.family == self.family
        assert self.dtype.bits == self.bits
        assert self.dtype == self.original


def _golden():
    for dtype, family, bits in _dtypes():
        yield _CaseChecker(original=dtype, family=family, bits=bits)


def _dtypes():
    yield "float16", "float", 16
    yield np.dtype("float16"), "float", 16
    yield torch.float16, "float", 16

    yield "float32", "float", 32
    yield np.dtype("float32"), "float", 32
    yield torch.float32, "float", 32

    yield "float64", "float", 64
    yield np.dtype("float64"), "float", 64
    yield torch.float64, "float", 64

    yield "int8", "int", 8
    yield np.dtype("int8"), "int", 8
    yield torch.int8, "int", 8

    yield "int16", "int", 16
    yield np.dtype("int16"), "int", 16
    yield torch.int16, "int", 16

    yield "int32", "int", 32
    yield np.dtype("int32"), "int", 32
    yield torch.int32, "int", 32

    yield "int64", "int", 64
    yield np.dtype("int64"), "int", 64
    yield torch.int64, "int", 64

    yield "bool", "bool", 8
    yield np.dtype("bool"), "bool", 8
    yield torch.bool, "bool", 8


@pytest.fixture(params=_golden())
def golden(request: FixtureRequest) -> _CaseChecker:
    return request.param


def test_dtype_cases(golden: _CaseChecker):
    golden.check()
