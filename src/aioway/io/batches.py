# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import math
import typing
from typing import TypeIs

from tensordict import TensorDict

from aioway.errors import AiowayError

from .frames import Frame

__all__ = ["BatchFrame", "BatchListFrame"]


@dcls.dataclass(frozen=True)
class BatchFrame(Frame, key="BATCH"):
    """
    A ``Frame`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    data: TensorDict
    """
    The underlying data of the ``Frame``.
    """

    batch: int
    """
    The batch size to use.
    """

    drop_last: bool = False
    """
    Whether to truncate the last batch that doesn't have length ``batch_size``.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.data, TensorDict):
            raise BatchTensorFrameTypeError(
                f"Expected {type(self)=} to be `TensorDict`."
            )

    @typing.override
    def __len__(self) -> int:
        truncate = math.ceil if self.drop_last else math.floor
        return truncate(len(self.data) / self.batch)

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        start = idx * self.batch
        end = min(start + self.batch, len(self.data))
        return self.data[start:end]

    @property
    def device(self):
        return self.data.device


@typing.final
@dcls.dataclass(frozen=True)
class BatchListFrame(Frame, key="LIST"):
    """
    A ``Frame`` backed by a ``list[TensorDict]`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    data: list[TensorDict]
    """
    The ``list`` of ``TensorDict``s.
    The data must all have the same keys and data types (#100),
    but not necessarily the same ``batch_size``.
    """

    def __post_init__(self) -> None:
        if not is_list_of_tensordict(self.data):
            raise BatchListTensorFrameTypeError(
                f"Expected a list of `TensorDict`s. Got {self.data=}"
            )

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def __getitem__(self, idx: int) -> TensorDict:
        return self.data[idx]

    def __setitem__(self, idx: int, value: TensorDict) -> None:
        self.data[idx] = value

    def append(self, td: TensorDict, /) -> None:
        self.data.append(td)

    def pop(self) -> TensorDict:
        return self.data.pop()


def is_list_of_tensordict(data) -> TypeIs[list[TensorDict]]:
    return isinstance(data, list) and all(isinstance(t, TensorDict) for t in data)


class BatchTensorFrameTypeError(AiowayError, TypeError): ...


class BatchListTensorFrameTypeError(AiowayError, TypeError): ...
