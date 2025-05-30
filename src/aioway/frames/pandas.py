# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

import torch
from pandas import DataFrame
from tensordict import TensorDict

from aioway.attrs import AttrSet, NamedAttr

from .frames import Frame

__all__ = ["PandasFrame"]


@dcls.dataclass(frozen=True)
class PandasFrame(Frame):
    df: DataFrame
    """
    The backing pandas dataframe.
    """

    device: str
    """
    The device to cast the tensors to.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.df)

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        row = self.df.iloc[idx].to_dict()
        return TensorDict(row).to(self.device)

    @typing.override
    def __getitems__(self, idx: list[int]) -> TensorDict:
        rows = self.df.iloc[idx].to_dict()
        tensors = {key: torch.tensor(val) for key, val in rows.items()}
        return TensorDict(tensors).to(self.device)

    @property
    @typing.override
    def attrs(self) -> AttrSet:
        # Note:
        #   No need for different devices because `pandas` stores on cpu.
        #   No need for shape check because `pandas` dataframes stores scalars.
        return AttrSet.from_iterable(
            NamedAttr.parse(
                {"name": name, "dtype": dtype, "shape": (), "device": self.device}
            )
            for name, dtype in zip(self.df.columns, self.df.dtypes)
        )
