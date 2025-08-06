# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
import logging
import typing
from abc import ABC
from typing import Any

import numpy as np
from numpy import dtype as NumpyDType
from torch import dtype as TorchDType

__all__ = ["DType"]

LOGGER = logging.getLogger(__name__)


@dcls.dataclass(eq=False, frozen=True)
class DType(ABC):
    """
    ``DType`` represents ``aioway``'s internal primitive data types,
    which is most similar to ``numpy.dtype`` and ``torch.dtype``.

    """

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        The ``str`` representation of the object.

        Guranteed to be ``numpy`` compatible.
        """

        ...

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

        LOGGER.debug("Computing %s == %s", self, other)

        from .factories import UnsupportedDTypeError

        # Try converting into something we know, if it fails,
        # leave it to `other` to implement.
        try:
            parsed = self.parse(other)
        except UnsupportedDTypeError:
            return NotImplemented

        # This would only happen if `other` is `None`.
        if parsed is None:
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
        """
        Parse the given dtype-like object, and return a ``DType`` instance.

        If the dtype cannot be parsed by the ``DTypeFactory``,
        ``NotImplemented`` is returned.
        """

        LOGGER.debug("Parsing %s", dtype)

        from .factories import DTypeFactory

        if dtype is None:
            return None

        # In case it's a `DTypeLike`, convert it to `DType` and use the same `__eq__`.
        factory = DTypeFactory()
        return factory(dtype)
