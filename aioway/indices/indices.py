# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from typing import ClassVar, Generic

from torch import Tensor

from aioway.schemas import ColumnSchema

__all__ = ["Index"]


class Index[T](ABC):
    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def dims(self) -> int: ...

    @abc.abstractmethod
    def search(self, query: T, k: int) -> Tensor: ...

    @classmethod
    @abc.abstractmethod
    def schema(cls) -> ColumnSchema:
        """
        The schema that the current index type can be indexed on.

        Returns:
            A schema.

        Todo:
            Extends it to handle multiple schema types,
            or a family of schemas.

            For example, a faiss index should be able to handle vector types,
            regardless of the dimension of the vector.
        """

        ...
