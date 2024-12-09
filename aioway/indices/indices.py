# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC

from torch import Tensor

__all__ = ["Index"]


class Index(ABC):
    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def train(self, data: Tensor) -> None: ...

    @abc.abstractmethod
    def dims(self) -> int: ...

    @abc.abstractmethod
    def search(self, query: Tensor, k: int) -> Tensor: ...

    @property
    def ndim(self) -> int:
        return self.dims()
