# Copyright (c) AIoWay Authors - All Rights Reserved

import numpy as np
from numpy.typing import DTypeLike
from torch import dtype as TorchDType

from aioway._errors import AiowayError

from .dtypes import DType
from .primitives import BoolDType, FloatDType, IntDType

__all__ = ["DTypeFactory"]


class DTypeFactory:
    """
    The factory for ``DType``.
    It leverages ``numpy``'s string support to construct different dtypes effectively.
    """

    def __call__(self, dtype: DTypeLike | DType | TorchDType, /) -> DType:
        """
        You call a ``DTypeFactory`` to generate a dtype from a ``DTypeLike`` or a ``DType``.

        Args:
            dtype: The arguemnt.

        Raises:
            UnsupportedDTypeError: If the type is not supported.

        Returns:
            The ``DType`` instance.
        """

        if isinstance(dtype, DType):
            return dtype

        # This is a little hacky since I'm a little lazy to implement
        # a lookup table with all `torch`'s `dtype`s.
        if isinstance(dtype, TorchDType):
            return self(str(dtype).removeprefix("torch."))

        try:
            dt = np.dtype(dtype)
        except TypeError as te:
            raise UnsupportedDTypeError(
                f"Type: {dtype} is not `DTypeLike`, cannot be converted."
            ) from te

        if np.isdtype(dt, "integral"):
            return IntDType(dt.itemsize * 8)

        if np.isdtype(dt, "real floating"):
            return FloatDType(dt.itemsize * 8)

        if np.isdtype(dt, "bool"):
            return BoolDType()

        raise UnsupportedDTypeError(f"{dt} is not a supported `numpy.dtype` kind.")


class UnsupportedDTypeError(AiowayError, ValueError, NotImplementedError): ...
