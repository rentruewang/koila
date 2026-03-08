# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway.ops import ParamList, Signature


def _signature_str():
    yield "(int, int) -> int"
    yield "(float, float) -> float"
    yield "(int, float) -> bool"


@pytest.fixture(params=_signature_str())
def signature(request):
    return Signature.parse(request.param, int=int, float=float, bool=bool)


def test_signature_param(signature):
    assert len(signature.params) == 2


@pytest.fixture
def param_list():
    return ParamList(int, int, float)


def test_param_list_compat(param_list):
    assert param_list.check(1, 2, 3.0)
    assert not param_list.check(1.0, 2, 3.0)
