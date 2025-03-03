# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Callable

import numpy as np
from numpy import ndarray as ArrayType
from numpy.typing import NDArray
from tensordict import TensorDict

from aioway.errors import AiowayError

from .batches import BatchStream, IterBatchStream

__all__ = ["FilterExec"]


@dcls.dataclass(frozen=True)
class FilterExec:
    predicate: Callable[[TensorDict], NDArray]

    def __call__(self, batches: BatchStream) -> BatchStream:
        return IterBatchStream(columns=batches.keys(), generator=self._gen(batches))

    def _gen(self, batches: BatchStream, /):
        for batch in batches.iterator():
            yield self._filter(batch)

    def _filter(self, data: TensorDict) -> TensorDict:
        keep = self.predicate(data)

        if not isinstance(keep, ArrayType):
            raise FilterIndexError(f"Index must be a numpy array. Got {type(keep)=}")

        if keep.ndim != 1:
            raise FilterIndexError(f"Index must be 1D, got {keep.ndim}D")

        # If dtype is boolean, ``True`` = keep, ``False`` = dicard.
        if np.issubdtype(keep.dtype, np.bool):
            if len(keep) != len(data):
                raise FilterIndexError(
                    "Boolean index must be of the same length as the tensordict."
                )

            return data[keep]

        # If dtype is int, The integers indicate the indices to keep.
        if np.issubdtype(keep.dtype, np.integer):
            if len(keep) >= len(data) or keep.min() < 0 or keep.max() >= len(data):
                raise FilterIndexError(
                    "Integer index should be valid and non repeating."
                )

            if len(np.unique(keep)) != len(keep):
                raise FilterIndexError("Integer index must be unique.")

            keep = np.sort(keep)
            return data[keep]

        raise FilterIndexError(f"Invalid index dtype: {keep.dtype}")


class FilterIndexError(AiowayError, IndexError): ...
