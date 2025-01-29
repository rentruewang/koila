# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import KeysView, Mapping
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from tensordict import TensorDict

from aioway.buffers import Buffer
from aioway.typings import Castable, Caster, Slicer

__all__ = ["Block", "BlockKind"]


type BlockKind = Literal["pandas", "tensordict"]


class Block(Castable, ABC):
    """
    ``Block`` represents a batch that is immutable,
    while providing some additional functionality.

    As a ``Block`` symbols a batch of data, selected from the dataset,
    it must be able to be converted to different formats like ``pandas`` or ``tensordict``.
    """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @typing.overload
    def __getitem__(self, idx: str) -> Buffer: ...

    @typing.overload
    def __getitem__(self, idx: int) -> Mapping[str, Any]: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[str] | list[int] | NDArray) -> Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._getitem_str(idx)

        if isinstance(idx, int):
            return self._getitem_int(idx)

        # Normalize the slices before passing in.
        if isinstance(idx, slice):
            slicer = Slicer(len(self))
            idx = slicer(idx)
            return self._getitem_slice(idx)

        # Columns must be a subset of the existing ones.
        if isinstance(idx, list) and all(isinstance(i, str) for i in idx):
            return self._getitem_cols(idx)

        # Other are ``ArrayLike``.
        idx = np.array(idx)
        return self._getitem_array(idx)

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool: ...

    @abc.abstractmethod
    def _getitem_str(self, idx: str) -> Buffer: ...

    @abc.abstractmethod
    def _getitem_int(self, idx: int) -> Mapping[str, Any]: ...

    @abc.abstractmethod
    def _getitem_slice(self, idx: slice) -> Self: ...

    @abc.abstractmethod
    def _getitem_array(self, idx: list[int] | NDArray) -> Self: ...

    @abc.abstractmethod
    def _getitem_cols(self, idx: list[str]) -> Self: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @abc.abstractmethod
    def sort_values(self, columns: list[str]) -> Self: ...

    @abc.abstractmethod
    def chain(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def zip(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def tensordict(self) -> TensorDict: ...

    @abc.abstractmethod
    def pandas(self) -> DataFrame: ...

    @classmethod
    def _caster(cls) -> Caster:
        from .pandas import PandasBlock
        from .torch import TensordictBlock

        def pandas_to_tensor(blk: PandasBlock):
            return TensordictBlock(blk.tensordict())

        def tensor_to_pandas(blk: TensordictBlock):
            return PandasBlock(blk.pandas())

        return Caster(
            base=Block,
            aliases=["tensordict", "pandas"],
            klasses=[TensordictBlock, PandasBlock],
            matrix=[
                [None, tensor_to_pandas],
                [pandas_to_tensor, None],
            ],
        )
