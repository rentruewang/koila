# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from numpy import typing as npt

from .indices import Index
from .ops import IndexAnn, IndexPlan

if typing.TYPE_CHECKING:
    import faiss


__all__ = ["FaissIndex"]


@dcls.dataclass(frozen=True)
class FaissIndex(Index):
    """
    The `Index` backed by the `faiss` library.
    """

    index: faiss.Index
    """
    The faiss index.
    """

    @typing.override
    def search(self, operator: IndexPlan, value: npt.NDArray) -> npt.NDArray:
        assert isinstance(operator, IndexAnn)

        if value.ndim != 2:
            raise ValueError(f"Value must be 2 dimensions. Got {value.ndim=}.")

        _, indices = self.index.search(value, operator.k)
        return indices

    @property
    @typing.override
    def dims(self) -> tuple[int]:
        return (self.index.d,)
