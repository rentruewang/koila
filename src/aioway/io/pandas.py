# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import math
import typing
from pathlib import Path
from typing import Any, Self

import pandas as pd
from pandas import DataFrame
from tensordict import TensorDict

from aioway.errors import AiowayError

from .frames import Frame

__all__ = ["PandasFrame"]


@dcls.dataclass(frozen=True)
class PandasFrame(Frame, key="PANDAS"):
    """
    A ``pandas``-based ``Frame``,
    dynamically converting ``DataFrame`` to ``TensorDict``.

    Todo:
        This class is essentially a duplicate of the ``BatchFrame`` class.
        See how to merge the logics / data together.
    """

    df: DataFrame
    """
    The backing ``DataFrame``.
    """

    batch_size: int
    """
    The batch size to use.
    """

    drop_last: bool = False
    """
    Whether to drop the last batch.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.df, DataFrame):
            raise PandasDataFrameTypeError(
                f"Expected pandas dataframe, got {type(self.df)=}"
            )

    @typing.override
    def __len__(self) -> int:
        rnd = math.floor if self.drop_last else math.ceil
        return rnd(len(self.df) / self.batch_size)

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.df))
        df = self.df.iloc[start:end]
        return TensorDict(df.to_dict("list"))

    @classmethod
    def read_csv(cls, csv: str | Path, **kwargs: Any) -> Self:
        df = pd.read_csv(csv)
        return cls(df=df, **kwargs)


class PandasDataFrameTypeError(AiowayError, TypeError): ...
