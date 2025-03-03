# Copyright (c) RenChu Wang - All Rights Reserved

import typing
from typing import Any, Literal

import deprecated as dprc
import numpy as np
from torch import Tensor

from aioway.errors import UnknownTypeError

from .buffers import Buffer
from .numpy import NumpyBuffer
from .torch import TorchBuffer

__all__ = ["buffer"]


@typing.overload
def buffer(data: Any, kind: Literal["numpy"]) -> NumpyBuffer: ...


@typing.overload
def buffer(data: Any, kind: Literal["torch"]) -> TorchBuffer: ...


@dprc.deprecated(reason="See issue #16")
def buffer(data, kind):
    buffer = _as_native_buffer(data=data)

    match kind:
        case "numpy":
            return buffer.numpy()
        case "torch":
            return buffer.torch()

    raise UnknownTypeError(
        f"Kind: {kind} is not known. Must be either 'numpy' or 'torch'."
    )


def _as_native_buffer(data) -> Buffer:
    if isinstance(data, Tensor):
        return TorchBuffer(data=data)

    data = np.array(data)
    return NumpyBuffer(data=data)
