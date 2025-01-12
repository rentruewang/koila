# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from collections.abc import KeysView
from typing import Self

import numpy as np
from numpy.typing import NDArray

from aioway.errors import AiowayError

__all__ = ["Block"]


@dcls.dataclass(frozen=True)
class Block[C, R](ABC):
    """
    ``Block`` is a thin wrapper over ``TensorDict``,
    while providing some additional functionality.
    """

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @typing.overload
    def __getitem__(self, idx: str) -> C: ...

    @typing.overload
    def __getitem__(self, idx: int) -> R: ...

    @typing.overload
    def __getitem__(self, idx: slice | list[str] | list[int] | NDArray) -> Self: ...

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self._getitem_str(idx)

        if isinstance(idx, int):
            return self._getitem_int(idx)

        if isinstance(idx, slice):
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
    def _getitem_str(self, idx: str) -> C: ...

    @abc.abstractmethod
    def _getitem_int(self, idx: int) -> R: ...

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
