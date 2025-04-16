# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
import typing
from typing import Self

from numpy.typing import NDArray

from aioway.errors import AiowayError
from aioway.execs import DataLoaderAdaptor, DataLoaderAdaptorLike

from .indices import Index, IndexContext
from .ops import IndexAnn, IndexOp

if typing.TYPE_CHECKING:
    from faiss import Index as FaissIdx


__all__ = ["FaissIndex"]


# TODO GPU support.
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

    @classmethod
    def create(
        cls,
        *,
        ctx: IndexContext,
        dl_opts: DataLoaderAdaptorLike = DataLoaderAdaptor(),
        factory: str
    ) -> Self:
        import faiss

        arr = cls.load_frame(ctx=ctx, dl_opts=dl_opts)

        # Create and train the index.
        index = faiss.index_factory(arr.shape[1], factory)
        index.train(arr)

        return cls(ctx=ctx, index=index)


class FaissIndexShapeError(AiowayError, IndexError): ...
