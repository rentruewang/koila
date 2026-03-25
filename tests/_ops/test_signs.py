# Copyright (c) AIoWay Authors - All Rights Reserved

from typing import Any

import pytest
from pytest import FixtureRequest

from aioway._signs import Signature, TypeList


def _signature_str():
    yield "(int, int) -> int"
    yield "(float, float) -> float"
    yield "(int, float) -> bool"


@pytest.fixture(params=_signature_str())
def signature(request: FixtureRequest):
    return Signature.parse(request.param, int=int, float=float, bool=bool)


def test_signature_param(signature: Signature[Any]):
    assert len(signature.param_types) == 2


@pytest.fixture
def type_list() -> TypeList:
    return TypeList(int, int, float)
