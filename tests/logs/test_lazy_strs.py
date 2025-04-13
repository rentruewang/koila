# Copyright (c) RenChu Wang - All Rights Reserved

import json
from pathlib import Path

import pytest

from aioway.logs import LazyStr


def strings():
    file = Path(__file__).parent / "strings.json"
    data = json.load(file.open())
    yield from data


@pytest.fixture(scope="module", params=strings())
def string(request) -> str:
    return request.param


def test_lazy_str_str(string):
    ls = LazyStr(lambda: string)
    assert str(ls) == string


def test_lazy_str_repr(string):
    ls = LazyStr(lambda: string)
    assert repr(ls) == string


def test_lazy_str_call(string):
    ls = LazyStr(lambda: string)
    assert ls() == string


def test_lazy_str_eq_str(string):
    ls = LazyStr(lambda: string)
    assert ls == string


def test_lazy_str_eq_obj(string):
    ls = LazyStr(lambda: string)
    assert ls == ls
