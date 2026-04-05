# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest
import tensordict as td

from aioway.fn import TensorDictFn, TensorFn, tdict


@pytest.fixture
def tdict_data():
    return td.TensorDict({"a": [1, 2, 3], "b": [4, 5, 6]}).auto_batch_size_()


@pytest.fixture
def tdict_data_fn(tdict_data: td.TensorDict):
    return tdict(tdict_data)


def _select_keys():
    yield list("a")
    yield list("b")
    yield list("ab")
    yield list("ba")


@pytest.fixture(params=_select_keys())
def select_keys(request):
    return request.param


def test_select(
    tdict_data_fn: TensorDictFn, tdict_data: td.TensorDict, select_keys: list[str]
):
    result = tdict_data_fn[select_keys].do()
    assert (result == tdict_data.select(*select_keys)).all()


def test_keys(tdict_data_fn: TensorDictFn):
    assert set(tdict_data_fn.keys()) == {"a", "b"}


def test_getitem(tdict_data_fn: TensorDictFn):
    assert isinstance(tdict_data_fn["a"], TensorFn)


def test_getitem_fail(tdict_data_fn: TensorDictFn):
    assert "g" not in tdict_data_fn.keys()
    with pytest.raises(KeyError):
        tdict_data_fn["g"]
