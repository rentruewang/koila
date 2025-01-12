# Copyright (c) RenChu Wang - All Rights Reserved

import pytest
from pandas import DataFrame
from tensordict import TensorDict

from aioway.blocks import TensorBlock


@pytest.fixture
def dataframe():
    return DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12]})


@pytest.fixture
def tensor_dict(dataframe):
    return TensorDict(dataframe.to_dict("series"))


@pytest.fixture
def dataframe_2():
    return DataFrame({"b": [1, 3, 5, 7, 9, 11], "c": [2, 4, 6, 8, 10, 12]})


@pytest.fixture
def left_block(dataframe):
    return TensorBlock.from_pandas(dataframe)


@pytest.fixture
def right_block(dataframe_2):
    return TensorBlock.from_pandas(dataframe_2)


def test_is_block(left_block):
    assert isinstance(left_block, TensorBlock)


def test_block_init(tensor_dict, left_block):

    assert isinstance(left_block.data, TensorDict)
    assert (left_block.data == tensor_dict).all()


def test_block_arithmetic(left_block):
    assert (left_block + left_block == left_block * 2).all()
    assert (left_block - left_block == 0).all()
    assert (left_block * 2 == left_block * 4 / 2).all()
    assert (left_block * 2 / 2 == left_block / 1).all()


def test_block_map(left_block):
    def cd(td: TensorDict) -> TensorDict:
        td = td.clone()

        td["c"] = td["a"] + td["b"]
        td["d"] = td["a"] - td["b"]

        return td

    assert (left_block.map(cd).data == cd(left_block.data)).all()


def test_block_zip(dataframe, left_block):
    assert list(dataframe.columns) == list("ab")

    col_a = dataframe["a"]
    col_b = dataframe["b"]

    block_a = TensorBlock.from_pandas(DataFrame({"a": col_a}))
    block_b = TensorBlock.from_pandas(DataFrame({"b": col_b}))

    assert (block_a.zip(block_b) == left_block).all()


def test_block_join(left_block, right_block):
    result = left_block.join(right_block, on="b")
    answer = {"b": [7, 9, 11], "a": [1, 3, 5], "c": [8, 10, 12]}

    result_in_pandas = result.to_pandas().sort_index(axis=1).sort_values("a")
    answer_in_pandas = DataFrame(answer).sort_index(axis=1).sort_values("a")

    result_in_pandas = result_in_pandas.reset_index(drop=True)
    answer_in_pandas = answer_in_pandas.reset_index(drop=True)

    assert (result_in_pandas == answer_in_pandas).all().all()
