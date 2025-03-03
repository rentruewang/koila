# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls

import deprecated as dprc
from numpy.typing import NDArray
from torch import Tensor

from .buffers import Buffer

__all__ = ["TorchBuffer"]


@dprc.deprecated(reason="See issue #16")
@dcls.dataclass(frozen=True)
class TorchBuffer(Buffer):

    data: Tensor
    """
    The underlying data for the ``TensorBuffer`` type.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, Tensor):
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
        return self.data

    def numpy(self) -> NDArray:
        return self.data.cpu().numpy()

    _getitem_array = _getitem_slice = __getitem
