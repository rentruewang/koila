# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Mapping
from typing import Any, Self

import deprecated as dprc
import numpy as np
import pandas as pd
from pandas import DataFrame
from tensordict import TensorDict

from aioway.buffers import NumpyBuffer

from .blocks import Block

__all__ = ["PandasBlock"]


@dprc.deprecated(reason="See issue #16")
@dcls.dataclass(frozen=True)
class PandasBlock(Block):
    data: DataFrame
    """
    The underlying data for the ``PandasBlock`` class.
    """

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key) -> bool:
        return key in self.data.keys()

    def keys(self):
        return self.data.keys()

    def sort_values(self, columns: list[str]) -> Self:
        return type(self)(self.data.sort_values(columns))

    def chain(self, other: Self) -> Self:
        return type(self)(pd.concat([self.data, other.data], axis=0))

    def zip(self, other: Self) -> Self:
        return type(self)(pd.concat([self.data, other.data], axis=1))

    def _getitem_str(self, idx: str) -> NumpyBuffer:
        return NumpyBuffer(np.array(self.data[idx]))

    def _getitem_int(self, idx: int) -> Mapping[str, Any]:
        return self.data.iloc[idx : idx + 1].to_dict()

    def _getitem_cols(self, idx: list[str]) -> Self:
        return type(self)(self.data[idx])

    def __getitem_iloc(self, idx):
        return type(self)(self.data.iloc[idx])

    _getitem_array = _getitem_slice = __getitem_iloc

    def pandas(self) -> DataFrame:
        return self.data

    def tensordict(self) -> TensorDict:
        return TensorDict(self.data.to_dict())
