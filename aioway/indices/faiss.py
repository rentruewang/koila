# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import Self

import faiss
from faiss import Index as FaissIndexType
from torch import Tensor

from .indices import Index

__all__ = ["FaissIndex"]


@dcls.dataclass(frozen=True)
class FaissIndex(Index):
    index: FaissIndexType

    def __len__(self):
        return self.index.ntotal

    def train(self, data: Tensor) -> None:
        self.index.train(data)

    def dims(self) -> int:
        return self.index.d

    def search(self, query: Tensor, k: int) -> Tensor:
        return self.index.search(query, k=k)

    @classmethod
    def index_factory(cls, dims: int, index_name: str) -> Self:
        index = faiss.index_factory(dims, index_name)
        return cls(index=index)
