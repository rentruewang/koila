# Copyright (c) AIoWay Authors - All Rights Reserved

"The ``Table`` interface."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from typing import Any, TypeAlias, TypeIs

import numpy as np
from numpy import ndarray as NDArrayType
from numpy.typing import NDArray
from tensordict import TensorDict
from torch.utils.data import Dataset

__all__ = ["Table", "IntArray", "TableDataset"]

IntArray: TypeAlias = NDArray[np.int_]
"Integer numpy array."

TableIndex: TypeAlias = slice | list[int] | IntArray
"The types that can be used for index accessing on ``Table``s."


@dcls.dataclass(frozen=True)
class Table(ABC):
    """
    ``Table`` represents a set of heterogenious data stored in memory,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    Think of it as a normal ``Sequence`` of ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Each ``TensorDict`` retrieved from ``Table`` is a minibatch of data.

    Similar to ``Dataset``, but only allows retrieving a batch at a time.
    To get a single item, retrieve a batch of size 1.

    For simplicity of API, this class does not support ``__getitem__(int)``,
    as that is not needed because all index access should be batched (slice, arrays),
    and unecessarily makes implementation duplicate.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

    def __getitem__(self, idx: slice | list[int] | IntArray, /) -> TensorDict:
        """
        Get individual items from the current ``Table``.

        Args:
            idx:
                Index to the current ``Frame``.
                Must be a slice, or list of int, or a numpy array.
                Should be in the range ``[-len, len)``.

        Returns:
            A ``TensorDict`` representing a batch of data.
        """

        if not _is_table_index(idx):
            raise IndexError(f"Index type {type(idx)=} is not supported.")

        # If slice, convert to ``range(len(self))[idx]``.
        # This will be the same length as the output list,
        # so it's ok that ``NDArray`` is less efficient than ``slice``.
        it: range | list[int] | IntArray
        if isinstance(idx, slice):
            it = range(len(self))[idx]
        else:
            it = idx

        arr: IntArray = np.asarray(it)
        arr = self._check_idx(arr)
        return self._getitem(arr)

    def __bool__(self) -> bool:
        return bool(len(self))

    @abc.abstractmethod
    def _getitem(self, idx: IntArray, /) -> TensorDict:
        """
        The implementation of ``__getitem__``.

        Args:
            idx: The index being passed in. A list of positive integers.

        Returns:
            A couple of rows i nthe dataset.
        """

        ...

    def _check_idx(self, idx: IntArray, /) -> IntArray:
        "Check if the index is valid, and then remap the index to be positive."

        length = len(self)

        if np.all(idx < -length) or np.all(idx >= length):
            raise IndexError(
                f"Index must be in the range `[-{length}, {length})`, but got {idx=}"
            )

        return idx % length

    def dataset(self) -> "TableDataset":
        return TableDataset(self)


@dcls.dataclass(frozen=True)
class TableDataset(Dataset[TensorDict]):
    "A ``Dataset`` adaptor that backs a ``Table``."

    table: Table

    def __len__(self) -> int:
        return len(self.table)

    @typing.no_type_check
    def __getitem__(self, idx: int) -> TensorDict:
        return self.table[[idx]][0]

    @typing.no_type_check
    def __getitems__(self, indices: list[int]) -> list[TensorDict]:
        return list(self.table[indices])


def _is_table_index(idx: Any) -> TypeIs[TableIndex]:
    "Check if the ``idx`` passed in is a valid type."

    # Check if it's a valid slice.
    if (
        True
        and isinstance(idx, slice)
        and isinstance(idx.stop, int)
        and isinstance(idx.start, int | None)
        and isinstance(idx.step, int | None)
    ):
        return True

    # Check if it's a ``list[int]``.
    if isinstance(idx, list) and all(isinstance(i, int) for i in idx):
        return True

    # Check if it's a ``NDArray[int]``.
    if isinstance(idx, NDArrayType) and np.isdtype(idx.dtype, "integral"):
        return True

    return False
