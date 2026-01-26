# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

import pytest

from aioway import attrs
from aioway.attrs import ComposedDType, DType, NumpyDType, TorchDType


@dcls.dataclass(frozen=True)
class _CaseChecker:
    dtype: DType
    family: str
    bits: int
    klass: type[DType]

    def check(self):
        assert isinstance(self.dtype, self.klass)
        assert self.dtype.family == self.family
        assert self.dtype.bits == self.bits


def _golden():
    for kind, klass in _kinds():
        for dtype, family, bits in _dtypes():
            yield _CaseChecker(
                attrs.dtype(dtype, kind=kind),
                family=family,
                bits=bits,
                klass=klass,
            )


def _kinds():
    yield "composed", ComposedDType
    yield "numpy", NumpyDType
    yield "torch", TorchDType


def _dtypes():
    yield "float16", "float", 16
    yield "float32", "float", 32
    yield "float64", "float", 64
    yield "int16", "int", 16
    yield "int32", "int", 32
    yield "int64", "int", 64
    yield "bool", "bool", 8


@pytest.fixture(params=_golden())
def golden(request):
    return request.param


def test_dtype_cases(golden: _CaseChecker):
    golden.check()
