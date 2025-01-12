# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import Self

import pandas as pd
from pandas import DataFrame, Series

from .blocks import Block


@dcls.dataclass(frozen=True)
class DataFrameBlock(Block[Series, DataFrame]):
    data: DataFrame

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

    def _getitem_str(self, idx: str) -> Series:
        return self.data[idx]

    def _getitem_int(self, idx: int) -> DataFrame:
        return self.data.iloc[idx : idx + 1]

    def _getitem_cols(self, idx: list[str]) -> Self:
        return type(self)(self.data[idx])

    def __getitem_iloc(self, idx):
        return type(self)(self.data.iloc[idx])

    _getitem_array = _getitem_slice = __getitem_iloc
