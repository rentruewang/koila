# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

import deprecated as dprc
import torch
from numpy import ndarray as ArrayType
from numpy.typing import NDArray
from torch import Tensor

from .buffers import Buffer

__all__ = ["NumpyBuffer"]


@dprc.deprecated(reason="See issue #16")
@dcls.dataclass(frozen=True)
class NumpyBuffer(Buffer):
    data: NDArray
    """
    The underlying data for the ``ArrayBuffer`` type.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, ArrayType):
            raise TypeError(
                f"Expected data to be of type Tensor, got {type(self.data)=}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem(self, idx):
        return type(self)(self.data[idx])

    def _size(self) -> tuple[int, ...]:
        return self.data.shape

    def torch(self) -> Tensor:
        return torch.from_numpy(self.data)

    def numpy(self) -> NDArray:
        return self.data

    _getitem_array = _getitem_slice = __getitem
