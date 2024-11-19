# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import pytest
from pandas import DataFrame
from tensordict import TensorDict

from aioway.blocks import Block


@pytest.fixture
def dataframe():
    return DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12]})


@pytest.fixture
def block(dataframe):
    return TensorDict(dataframe.to_dict("series"))


def test_block_init(dataframe, block):
    df = Block.from_pandas(dataframe)

    assert isinstance(df.data, TensorDict)
    assert (df.data == block).all()


def test_block_arithmetic(dataframe):
    df = Block.from_pandas(dataframe)
    assert (df + df == df * 2).all()
    assert (df - df == 0).all()
    assert (df * 2 == df * 4 / 2).all()
