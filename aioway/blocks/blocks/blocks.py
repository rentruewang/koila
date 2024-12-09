# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import typing
from abc import ABC
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Self

import pandas as pd
from pandas import DataFrame
from tensordict import TensorDict
from torch import Tensor

from aioway.blocks._typing import Primitive
from aioway.schemas import TableSchema

from .buffers import Buffer

__all__ = ["Block"]


class Block[B: Buffer](ABC):
    """
    ``Block`` represents a chunk / batch of data stored in memory,
    it is the main physical abstraction in ``aioway`` to represent eager computation.

    Think of it as a normal ``pandas.DataFrame`` or ``torch.Tensor`` or ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.
    """

    def __len__(self) -> int:
        return self.count()

    @abc.abstractmethod
    def __contains__(self, key: object) -> bool: ...

    @typing.overload
    def __getitem__(self, key: str) -> B: ...

    @typing.overload
    def __getitem__(self, key: list[str]) -> Self: ...

    @typing.overload
    def __getitem__(self, key: int) -> dict[str, Primitive]: ...

    @typing.overload
    def __getitem__(self, key: list[int] | slice | Tensor) -> Self: ...

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._col(key)

        if isinstance(key, int):
            return self._row(key)

        if isinstance(key, slice):
            return self._slice_of_rows(key)

        if isinstance(key, Tensor):
            return self._list_of_rows(key)

        if isinstance(key, list):
            # Note:
            #   Normally, here we would be using ``isinstance`` checks on each individual indices.
            #   However, doing that is very time consuming,
            #   and we would not want subclasses of ``int`` or ``str`` here anyways.
            #   For example, ``bool`` is a subclass of ``int``, but is undesired here.
            types = {type(i) for i in key}

            if types == {int}:
                return self._list_of_rows(key)

            if types == {str}:
                return self.project(*key)

            raise TypeError(f"List must be a list of `int`s or `str`. Got {types}")

        raise TypeError(f"{type(key)=} is not supported!")

    @abc.abstractmethod
    def _col(self, idx: str) -> B: ...

    @abc.abstractmethod
    def _row(self, idx: int) -> dict[str, Primitive]: ...

    @abc.abstractmethod
    def _slice_of_rows(self, idx: slice) -> Self: ...

    @abc.abstractmethod
    def _list_of_rows(self, idx: list[int] | Tensor) -> Self: ...

    @abc.abstractmethod
    def __neg__(self) -> Self: ...

    @abc.abstractmethod
    def __invert__(self) -> Self: ...

    @abc.abstractmethod
    def __eq__(self, other: Self) -> Self:  # type: ignore[override]
        ...

    @abc.abstractmethod
    def __ne__(self, other: Self) -> Self:  # type: ignore[override]
        ...

    @abc.abstractmethod
    def __ge__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __gt__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __le__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __lt__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __add__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __sub__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __mul__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __truediv__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __floordiv__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def __pow__(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def gather(self, dim: int, index: Sequence[int]) -> Self: ...

    @abc.abstractmethod
    def rename(self, **names: str) -> Self: ...

    @abc.abstractmethod
    def map(self, f: Callable[[TensorDict], TensorDict], /) -> Self: ...

    @abc.abstractmethod
    def reduce[I](self, f: Callable[[TensorDict, I], I], init: I) -> I: ...

    @abc.abstractmethod
    def zip(self, other: Self) -> Self: ...

    @abc.abstractmethod
    def join(self, other: Self, on: str) -> Self: ...

    def select(self, idx: list[int], /) -> Self:
        len_self = self.count()

        if idx and (min(idx) < -len_self or max(idx) >= len_self):
            out_of_bounds = [i for i in idx if i >= len_self or i < -len_self]
            raise IndexError(
                f"Index: {out_of_bounds} out of bounds for Block of length {len_self}."
            )

        return self._select(idx)

    @abc.abstractmethod
    def _select(self, idx: list[int], /) -> Self: ...

    def project(self, *cols: str) -> Self:
        if extras := set(cols).difference(names := self.schema().names):
            raise ValueError(
                f"Columns: {extras} specified, "
                f"but not found in schema's columns: {names}."
            )

        return self._project(*cols)

    @abc.abstractmethod
    def _project(self, *cols: str) -> Self: ...

    @abc.abstractmethod
    def schema(self) -> TableSchema:
        """
        The schema type from the current block.
        """

        ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def max(self) -> dict[str, Primitive]: ...

    @abc.abstractmethod
    def min(self) -> dict[str, Primitive]: ...

    @abc.abstractmethod
    def to_pandas(self) -> DataFrame: ...

    @abc.abstractmethod
    def to_tensordict(self) -> TensorDict: ...

    def to_csv(self, path: str | Path, /) -> None:
        df = self.to_pandas()
        return df.to_csv(path)

    @property
    def columns(self) -> list[str]:
        return self.schema().names

    @classmethod
    def from_csv(cls, csv: str | Path, /) -> Self:
        df = pd.read_csv(csv)
        return cls.from_pandas(df)

    @classmethod
    @abc.abstractmethod
    def from_pandas(cls, df: DataFrame, /) -> Self: ...
