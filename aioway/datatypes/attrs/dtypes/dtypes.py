# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
import typing
from abc import ABC
from typing import Any

import numpy as np
from numpy import dtype as NumpyDType
from torch import dtype as TorchDType

__all__ = ["DType"]


@dcls.dataclass(eq=False, frozen=True)
class DType(ABC):
    """
    ``DType`` represents ``aioway``'s internal primitive data types,
    which is most similar to ``numpy.dtype`` and ``torch.dtype``.

    Todo:
        Add a visitor for ``DType``.
    """

    @abc.abstractmethod
    def __str__(self) -> str: ...

    def __eq__(self, other: Any) -> bool:
        """
        The ``==`` method for ``DType``.

        Args:
            other:
                The RHS would be processed into a ``DType`` by using ``DTypeFactory``,
                which can be ``DTypeLike`` or ``DType``, including string support by ``numpy``.

        Returns:
            A boolean.
        """

        if (parsed := self.parse(other)) in [None, NotImplemented]:
            return False

        return self.numpy() == parsed.numpy()

    def numpy(self) -> NumpyDType:
        return np.dtype(str(self))

    @typing.overload
    @classmethod
    def parse(cls, dtype: "str | NumpyDType | DType | TorchDType") -> "DType": ...

    @typing.overload
    @classmethod
    def parse(cls, dtype: None) -> None: ...

    @classmethod
    def parse(cls, dtype):
        from .factories import DTypeFactory, UnsupportedDTypeError

        if dtype is None:
            return None

        # In case it's a ``DTypeLike``, convert it to ``DType`` and use the same ``__eq__``.
        factory = DTypeFactory()

        try:
            return factory(dtype)
        except UnsupportedDTypeError:
            return NotImplemented
