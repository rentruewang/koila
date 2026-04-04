# Copyright (c) AIoWay Authors - All Rights Reserved


import pytest
import tensordict as td

from aioway import fn


@pytest.fixture
def tdict():
    return td.TensorDict({"a": [1, 2, 3], "b": [4, 5, 6]}).auto_batch_size_()


@pytest.fixture
def tdict_fn(tdict: td.TensorDict):
    return fn.tdict(tdict)


def _select_keys():
    yield list("a")
    yield list("b")
    yield list("ab")
    yield list("ba")


@pytest.fixture(params=_select_keys())
def select_keys(request):
    return request.param


def test_select(
    tdict_fn: fn.TensorDictFn, tdict: td.TensorDict, select_keys: list[str]
):
    result = tdict_fn[select_keys].do()
    assert (result == tdict.select(*select_keys)).all()


def test_keys(tdict_fn: fn.TensorDictFn):
    assert set(tdict_fn.keys()) == {"a", "b"}


def test_getitem(tdict_fn: fn.TensorDictFn):
    assert isinstance(tdict_fn["a"], fn.TensorFn)


def test_getitem_fail(tdict_fn: fn.TensorDictFn):
    assert "g" not in tdict_fn.keys()
    with pytest.raises(KeyError):
        tdict_fn["g"]
