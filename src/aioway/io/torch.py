# Copyright (c) AIoWay Authors - All Rights Reserved

import dataclasses as dcls
import typing
from typing import TypeIs

from tensordict import TensorDict

from aioway._errors import AiowayError

from ._batches import BatchFrame
from .frames import Frame

__all__ = ["TorchFrame", "TorchListFrame"]


@dcls.dataclass(frozen=True)
class TorchFrame(BatchFrame, key="TORCH"):
    """
    A ``Frame`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    KLASS = TensorDict

    @classmethod
    @typing.override
    def convert_tensordict(cls, data: TensorDict) -> TensorDict:
        return data


@typing.final
@dcls.dataclass(frozen=True)
class TorchListFrame(Frame, key="LIST"):
    """
    A ``Frame`` backed by a ``list[TensorDict]`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    data: list[TensorDict] = dcls.field(default_factory=list)
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


class TensorFrameTypeError(AiowayError, TypeError): ...


class BatchListTensorFrameTypeError(AiowayError, TypeError): ...
