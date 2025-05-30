# Copyright (c) AIoWay Authors - All Rights Reserved

import functools
import json
from pathlib import Path

import numpy as np
import pytest

from aioway.attrs import DType, DTypeFactory


@functools.cache
def _supported_dtypes():
    with open(Path(__file__).parent / "dtypes.json") as f:
        return json.load(f)


def _raw_dtypes():
    yield from _supported_dtypes()["raw"]


def _canon_dtypes():
    yield from _supported_dtypes()["canon"]


def _all_dtypes():
    yield from _raw_dtypes()
    yield from _canon_dtypes()


@pytest.fixture(params=_raw_dtypes(), scope="module")
def raw_dtype(request) -> str:
    return request.param


@pytest.fixture(params=_all_dtypes(), scope="module")
def dtype(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def factory():
    return DTypeFactory()


def test_dtype_factory_generation_type(factory, dtype):
    assert isinstance(factory(dtype), DType)


def test_dtype_reprs_raw(factory, raw_dtype):
    assert str(factory(raw_dtype)) == raw_dtype


def test_dtype_eq_str(factory, dtype):
    assert factory(dtype) == dtype


def test_dtype_eq_np_dtype(factory, dtype):
    assert factory(dtype) == np.dtype(dtype)
