# Copyright (c) RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import KeysView
from typing import Self

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor
from torch import device as Device

from aioway.errors import AiowayError

from .blocks import Block

__all__ = ["TensorBlock"]


@dcls.dataclass(frozen=True)
class TensorBlock(Block[Tensor, TensorDict]):
    data: TensorDict
    """
    The underlying ``TensorDict`` that is treated as a batch.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDict):
            raise BatchTypeError(
                "Underlying data for `Batch` should be of type `TensorDict`, "
                f"sgot {type(self.data)=}"
            )

        if not len(self.data):
            raise NotBatchedError(f"TensorDict is not batched. {self.data.batch_size=}")

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def keys(self) -> KeysView[str]:
        return self.data.keys()

    def get(self) -> TensorDict:
        """
        Fetch the underlying ``TensorDict``.
        """

        return self.data

    def to(self, device: str | Device) -> Self:
        self._set_device(device)
        return self

    def sort_values(self, columns: list[str]) -> Self:
        # Reversed because numpy lexsort uses the last key as main key.
        cols = np.array([self.data[c].tolist() for c in reversed(columns)])
        orders = np.array(np.lexsort(cols))
        return self[orders]

    def chain(self, other: Self) -> Self:
        self._check_other_device(other)

        if self.keys() != other.keys():
            raise BatchChainError(
                "Batch keys must match to chain. "
                f"Got {self.keys()=} and {other.keys()=}"
            )

        return type(self)(torch.cat([self.data, other.data], dim=0))

    def zip(self, other: Self) -> Self:
        self._check_other_device(other)

        if len(self) != len(other):
            raise BatchZipError(
                "Can only zip batches with the same length. "
                f"Got {len(self)=} and {len(other)=}"
            )

        return type(self)(TensorDict({**self.data, **other.data}, device=self.device))

    def _check_other_device(self, other: Self, /) -> None:
        if self.device != other.device:
            raise BatchDeviceError(
                "Can only merge `TensorDict`s on the same device. "
                f"Got {self.device=} and {other.device=}"
            )

    def _getitem_str(self, idx: str) -> Tensor:
        return self.data[idx]

    def _getitem_cols(self, idx: list[str]) -> Self:
        return type(self)(self.data.select(*idx))

    def __getitem_direct(self, idx: object) -> Self:
        return type(self)(self.data[idx])

    _getitem_int = _getitem_slice = _getitem_array = __getitem_direct

    def _get_device(self) -> Device:
        return self.data.device or Device("cpu")

    def _set_device(self, device: str | Device, /) -> None:
        self.data.to(device=device)

    device = property(fget=_get_device, fset=_set_device)


class BatchTypeError(AiowayError, TypeError): ...


class NotBatchedError(AiowayError, ValueError): ...


class BatchChainError(AiowayError, ValueError): ...


class BatchZipError(AiowayError, ValueError): ...


class BatchDeviceError(AiowayError, ValueError): ...
