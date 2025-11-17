# Copyright (c) AIoWay Authors - All Rights Reserved

"``Table``s that produce data by slicing contiguous input records."

import abc
import dataclasses as dcls
import math
import typing
from abc import ABC

from pandas import DataFrame
from tensordict import TensorDict

from .tables import Table

__all__ = ["SourceTable", "TensorDictTable"]


@dcls.dataclass(frozen=True)
class SourceTable(Table, ABC):
    """
    A ``Table`` acting like a source, non-distributed, and volatile.
    """

    batch: int
    """
    The batch size to use.
    """

    device: str = "cpu"
    """
    The device to send the batches to.
    """

    drop_last: bool = False
    """
    Whether to truncate the last batch that doesn't have length ``batch_size``.
    """

    @typing.final
    @typing.override
    def __len__(self) -> int:
        truncate = math.ceil if self.drop_last else math.floor
        return truncate(self._num_records() / self.batch)

    @typing.final
    @typing.override
    def _getitem(self, idx: int, /) -> TensorDict:
        start = idx * self.batch
        end = min((idx + 1) * self.batch, self._num_records())
        result = self._slice(start, end)
        result = result.to(self.device)
        return result

    @abc.abstractmethod
    def _num_records(self) -> int:
        """
        Number of records in the underlying source.
        """

        ...

    @abc.abstractmethod
    def _slice(self, start: int, end: int, /) -> TensorDict: ...


@dcls.dataclass(frozen=True)
class _TensorDictMixin:
    source: TensorDict


@dcls.dataclass(frozen=True)
class TensorDictTable(SourceTable, _TensorDictMixin):
    """
    A ``Table`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    @typing.override
    def _num_records(self) -> int:
        return len(self.source)

    @typing.override
    def _slice(self, start: int, end: int, /) -> TensorDict:
        return self.source[start:end]


@dcls.dataclass(frozen=True)
class _PandasDataFrameMixin:
    source: DataFrame


@dcls.dataclass(frozen=True)
class PandasTable(SourceTable, _PandasDataFrameMixin):
    @typing.override
    def _num_records(self) -> int:
        return len(self.source)

    @typing.override
    def _slice(self, start: int, end: int, /) -> TensorDict:
        dicts = self.source[start:end].to_dict()
        return TensorDict(dicts)
