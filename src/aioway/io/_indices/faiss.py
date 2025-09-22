# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from numpy.typing import NDArray

from aioway._errors import AiowayError

from .indices import Index
from .ops import IndexAnn, IndexOp

if typing.TYPE_CHECKING:
    from faiss import Index as FaissIdx


__all__ = ["FaissIndex"]


@dcls.dataclass(frozen=True)
class FaissIndex(Index):
    """
    The ``Index`` backed by the ``faiss`` library.
    """

    index: "FaissIdx"
    """
    The faiss index.
    """

    @typing.override
    def search(self, operator: IndexOp, value: NDArray) -> NDArray:
        assert isinstance(operator, IndexAnn)

        if value.ndim != 2:
            raise FaissIndexShapeError("Value must be 2 dimensions.")

        _, indices = self.index.search(value, operator.k)
        return indices

    @property
    @typing.override
    def dims(self) -> tuple[int]:
        return (self.index.d,)


class FaissIndexShapeError(AiowayError, IndexError): ...
