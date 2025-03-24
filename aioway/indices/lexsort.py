# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import Self

import numpy as np
from numpy.typing import NDArray

from aioway.errors import AiowayError
from aioway.execs import DataLoaderAdaptor, DataLoaderAdaptorLike

from .indices import Index, IndexContext


@dcls.dataclass(frozen=True)
class LexsortIndex(Index):
    sorted_index: NDArray
    """
    The 1D index for which
    """

    def __post_init__(self) -> None:
        if self.sorted_index.ndim != 1:
            raise LexsortShapeError("Sorted index must be 1 dimensional.")

    @classmethod
    def create(
        cls,
        ctx: IndexContext,
        dl_opts: DataLoaderAdaptorLike = DataLoaderAdaptor(),
    ) -> Self:
        arr = cls.load_frame(ctx=ctx, dl_opts=dl_opts)

        # Reverse and sort s.t. ``lexsort`` would sort in the order of columns in ``arr``.
        rev = arr[:, ::-1]

        result = np.lexsort(rev)
        assert result.ndim == 1

        return cls(ctx=ctx, sorted_index=result)


class LexsortShapeError(AiowayError, ValueError): ...
