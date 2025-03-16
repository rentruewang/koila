# Copyright (c) RenChu Wang - All Rights Reserved

import functools
import json
from pathlib import Path

import numpy as np
import pytest

from aioway.datatypes import DType, DTypeFactory


@functools.cache
def supported_dtypes() -> list[str]:
    with open(Path(__file__).parent / "dtypes.json") as f:
        return json.load(f)


@pytest.fixture(params=supported_dtypes(), scope="module")
def dtype(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def factory():
    return DTypeFactory()


def test_dtype_factory_generation_type(factory, dtype):
    assert isinstance(factory(dtype), DType)


def test_dtype_reprs(factory, dtype):
    assert str(factory(dtype)) == dtype


def test_dtype_eq_str(factory, dtype):
    assert factory(dtype) == dtype


def test_dtype_eq_np_dtype(factory, dtype):
    assert factory(dtype) == np.dtype(dtype)
