# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing

from torch.nn import Embedding as _Embedding

from aioway.attrs import DType, DTypeLike, Shape, ShapeLike

from .previews import Preview


@dcls.dataclass(frozen=True)
class Embedding(Preview):
    MODULE_TYPE = _Embedding

    num_embeddings: int
    embedding_dim: int

    @typing.override
    def _preview_shape(self, shape: Shape, /) -> ShapeLike:
        return [*shape, self.embedding_dim]

    @typing.override
    def _preview_dtype(self, dtype: DType) -> DTypeLike:
        if dtype.family != "int":
            raise ValueError

        # If `dtype` is not specified, default to "float".
        return self.dtype or "float"
