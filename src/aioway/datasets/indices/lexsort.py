# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from numpy import typing as npt

from . import indices

__all__ = ["LexsortIndex"]


@dcls.dataclass(frozen=True)
class LexsortIndex(indices.Index):
    sorted_index: npt.NDArray
    """
    The 1D index for which
    """

    def __post_init__(self) -> None:
        if self.sorted_index.ndim != 1:
            raise ValueError("Sorted index must be 1 dimensional.")
