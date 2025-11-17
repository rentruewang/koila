# Copyright (c) AIoWay Authors - All Rights Reserved

"``Table``s are stored in memory."

import dataclasses as dcls
import typing
from typing import Self, TypeIs

from tensordict import TensorDict

from aioway.streams import Stream

from .tables import Table

__all__ = ["TensorDictListTable"]


@typing.final
@dcls.dataclass(frozen=True)
class TensorDictListTable(Table):
    """
    A ``Table`` backed by a ``list[TensorDict]`` (aka a batch in ``aioway``).
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
            raise ValueError(f"Expected a list of `TensorDict`s. Got {self.data=}")

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def _getitem(self, idx: int) -> TensorDict:
        return self.data[idx]

    def __setitem__(self, idx: int, value: TensorDict) -> None:
        self.data[idx] = value

    def append(self, td: TensorDict, /) -> None:
        self.data.append(td)

    def pop(self) -> TensorDict:
        return self.data.pop()

    @classmethod
    def consume(cls, stream: Stream) -> Self:
        """
        Consume a ``Stream`` and produce a ``Table``.

        Args:
            stream: The ``Stream``. Would be exhausted after calling.

        Returns:
            The ``Table`` produced.
        """

        dicts = list(stream)
        return cls(dicts)


def is_list_of_tensordict(data) -> TypeIs[list[TensorDict]]:
    return isinstance(data, list) and all(isinstance(t, TensorDict) for t in data)
