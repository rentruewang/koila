# Copyright (c) AIoWay Authors - All Rights Reserved

import typing

import pytest

from aioway import _signs


def _signature_str():
    yield "(int, int) -> int"
    yield "(float, float) -> float"
    yield "(int, float) -> bool"


@pytest.fixture(params=_signature_str())
def signature(request: pytest.FixtureRequest):
    return _signs.Signature.parse(request.param, int=int, float=float, bool=bool)


def test_signature_param(signature: _signs.Signature[typing.Any]):
    assert len(signature.param_types) == 2


@pytest.fixture
def type_list() -> _signs.TypeList:
    return _signs.TypeList(int, int, float)
