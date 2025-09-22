# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls

from numpy.typing import NDArray

from aioway._errors import AiowayError

from .indices import Index

__all__ = ["LexsortIndex"]


@dcls.dataclass(frozen=True)
class LexsortIndex(Index):
    sorted_index: NDArray
    """
    The 1D index for which
    """

    def __post_init__(self) -> None:
        if self.sorted_index.ndim != 1:
            raise LexsortShapeError("Sorted index must be 1 dimensional.")


class LexsortShapeError(AiowayError, ValueError): ...
