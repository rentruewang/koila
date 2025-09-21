# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from typing import TypeIs

import numpy as np
from numpy import ndarray as NDArrayType
from numpy.typing import NDArray
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

from aioway import registries
from aioway.errors import AiowayError

__all__ = ["Frame"]


@dcls.dataclass(frozen=True)
class Frame(Dataset[TensorDict], ABC):
    """
    ``Frame`` represents a chunk / batch of heterogenious data stored in memory,
    it is one of the main physical abstractions in ``aioway`` to represent eager computation.

    Think of it as a normal ``Sequence`` of ``TensorDict``,
    where computation happens eagerly, imperatively, and the result is stored in memory.

    Each ``TensorDict`` retrieved from ``Frame`` is a minibatch of data.
    """

    def __init_subclass__(cls, key: str):
        init_sublcass = registries.init_subclass(lambda: Frame)
        init_sublcass(cls, key=key)

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Get the number of items (rows) in the current dataframe.
        """

    def __getitem__(
        self, idx: int | slice | list[int] | NDArray | Tensor
    ) -> TensorDict:
        """
        Get individual items from the current ``Frame``.
        """

        if isinstance(idx, int):
            return self._getitem_int(idx)

        if isinstance(idx, slice):
            return self._getitem_slice(idx)

        if isinstance(idx, Tensor):
            return self._getitem_tensor(idx)

        if isinstance(idx, NDArrayType) or is_list_of_int(idx):
            idx = np.array(idx)
            return self._getitem_arr(idx)

        raise FrameIndexTypeError(
            f"Unrecognized {type(idx)=}. "
            "Should be one of `int`, `slice`, `list[int]`, `NDArray`, `Tensor`."
        )

    __getitems__ = __getitem__
    "``__getitems__`` is a torch specific API."

    @abc.abstractmethod
    def _getitem_int(self, idx: int, /) -> TensorDict: ...

    @abc.abstractmethod
    def _getitem_slice(self, idx: slice, /) -> TensorDict: ...

    @abc.abstractmethod
    def _getitem_arr(self, idx: NDArray, /) -> TensorDict: ...

    @abc.abstractmethod
    def _getitem_tensor(self, idx: Tensor, /) -> TensorDict: ...

    def __str__(self) -> str:
        return repr(self)


def is_list_of_int(obj) -> TypeIs[list[int]]:
    return isinstance(obj, list) and all(isinstance(i, int) for i in obj)


class FrameIndexTypeError(AiowayError, TypeError): ...
