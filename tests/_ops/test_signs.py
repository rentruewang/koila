# Copyright (c) AIoWay Authors - All Rights Reserved

from typing import Any

import pytest
from pytest import FixtureRequest

from aioway._ops import OpSign, TypeList


def _signature_str():
    yield "(int, int) -> int"
    yield "(float, float) -> float"
    yield "(int, float) -> bool"


@pytest.fixture(params=_signature_str())
def signature(request: FixtureRequest):
    return OpSign.parse(request.param, int=int, float=float, bool=bool)


def test_signature_param(signature: OpSign[Any]):
    assert len(signature.param_types) == 2


@pytest.fixture
def type_list() -> TypeList:
    return TypeList(int, int, float)


def test_param_list_compat(type_list: TypeList):
    assert type_list.check_values([1, 2, 3.0])
    assert not type_list.check_values([1.0, 2, 3.0])
