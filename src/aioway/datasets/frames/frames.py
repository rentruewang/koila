# Copyright (c) AIoWay Authors - All Rights Reserved

"The `Frame` interface."

import abc
import dataclasses as dcls
import typing
from abc import ABC
from typing import Any, TypeIs

import numpy as np
from numpy import ndarray as NpArr

from aioway import _typing
from aioway._previews import AttrSet
from aioway._typing import BatchIndex, IntArray
from aioway.chunks import Chunk

from ..datasets import Dataset, DatasetViewTypes

if typing.TYPE_CHECKING:
    from .views import FrameColumnView, FrameSelectView

__all__ = ["Frame"]


@dcls.dataclass(frozen=True)
class Frame(Dataset, ABC):
    """
    `Frame` represents a set of heterogenious data stored in memory,
    it is one of the main physical abstractions in `aioway` to represent eager computation.

    Think of it as a normal `Sequence` of `Chunk`,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Each `Chunk` retrieved from `Frame` is a minibatch of data.

    Similar to `Dataset`, but only allows retrieving a batch at a time.
    To get a single item, retrieve a batch of size 1.

    For simplicity of API, this class does not support `__getitem__(int)`,
    as that is not needed because all index access should be batched (slice, arrays),
    and unecessarily makes implementation duplicate.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

    @typing.overload
    def __getitem__(self, idx: BatchIndex, /) -> Chunk: ...

    @typing.overload
    def __getitem__(self, idx: str, /) -> FrameColumnView: ...

    @typing.overload
    def __getitem__(self, idx: list[str], /) -> FrameSelectView: ...

    @typing.no_type_check
    def __getitem__(self, idx, /):
        """
        Get individual items from the current `Frame`.

        Args:
            idx:
                Index to the current `Frame`.
                If it is a `str` or `list[str]`, it is considered a `Table` operation.

                For indexing operations, index type must be a `slice`,
                or `list[int]`, or a numpy array.
                Should be in the range `[-len, len)`.

        Returns:
            A `TensorDict` representing a batch of data.
        """

        if isinstance(idx, str):
            return self.column(idx)

        if _typing.is_list_of(str)(idx):
            return self.select(*idx)

        if not _is_table_index(idx):
            raise IndexError(f"Index type {type(idx)=} is not supported.")

        # If slice, convert to `range(len(self))[idx]`.
        # This will be the same length as the output list,
        # so it's ok that `NDArray` is less efficient than `slice`.
        it: range | list[int] | IntArray
        if isinstance(idx, slice):
            it = range(len(self))[idx]
        else:
            it = idx

        arr: IntArray = np.asarray(it)
        arr = self._check_idx(arr)

        if (item := self._getitem(arr)).attrs != self.attrs:
            raise ValueError(f"Attr mismatch for {item.attrs=} and {self.attrs=}.")

        return item

    def __bool__(self) -> bool:
        return bool(len(self))

    @property
    @abc.abstractmethod
    def attrs(self) -> AttrSet:
        "The schema of the current frame."

        raise NotImplementedError

    @abc.abstractmethod
    def _getitem(self, idx: IntArray, /) -> Chunk:
        """
        The implementation of `__getitem__`.

        Args:
            idx: The index being passed in. A list of positive integers.

        Returns:
            A couple of rows i nthe dataset.
        """

        raise NotImplementedError

    @classmethod
    @typing.override
    def view_types(cls):
        from .views import FrameColumnView, FrameSelectView

        return DatasetViewTypes(column=FrameColumnView, select=FrameSelectView)

    def _check_idx(self, idx: IntArray, /) -> IntArray:
        "Check if the index is valid, and then remap the index to be positive."

        length = len(self)

        if np.all(idx < -length) or np.all(idx >= length):
            raise IndexError(
                f"Index must be in the range `[-{length}, {length})`, but got {idx=}"
            )

        return idx % length


def _is_table_index(idx: Any) -> TypeIs[BatchIndex]:
    "Check if the `idx` passed in is a valid type."

    # Check if it's a valid slice.
    if (
        True
        and isinstance(idx, slice)
        and isinstance(idx.stop, int)
        and isinstance(idx.start, int | None)
        and isinstance(idx.step, int | None)
    ):
        return True

    # Check if it's a `list[int]`.
    if isinstance(idx, list) and all(isinstance(i, int) for i in idx):
        return True

    # Check if it's a `NDArray[int]`.
    if isinstance(idx, NpArr) and np.isdtype(idx.dtype, "integral"):
        return True

    return False
