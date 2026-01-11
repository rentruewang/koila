# Copyright (c) AIoWay Authors - All Rights Reserved

"``Table``s that produce data by slicing contiguous input records."

import dataclasses as dcls
import typing
from typing import TypeIs

import numpy as np
import tensordict
from tensordict import TensorDict

from .tables import IntArray, Table

__all__ = ["TensorDictTable", "TensorDictListTable"]


@dcls.dataclass(frozen=True)
class TensorDictTable(Table):
    """
    A ``Table`` backed by a ``TensorDict`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    data: TensorDict
    """
    The ``TensorDict`` source.
    """

    @typing.override
    def __len__(self) -> int:
        return len(self.data)

    @typing.override
    def _getitem(self, idx: IntArray) -> TensorDict:
        return self.data[idx]


@typing.final
@dcls.dataclass(frozen=True)
class TensorDictListTable(Table):
    """
    A ``Table`` backed by a ``list[TensorDict]`` (aka a batch in ``aioway``).
    This means that it is non-distributed, and volatile.
    """

    tensordicts: list[TensorDict] = dcls.field(default_factory=list)
    """
    The ``list`` of ``TensorDict``s.
    The data must all have the same keys and data types (#100),
    but not necessarily the same ``batch_size``.
    """

    def __post_init__(self) -> None:
        if not is_list_of_tensordict(self.tensordicts):
            raise ValueError(
                f"Expected a list of `TensorDict`s. Got {self.tensordicts=}"
            )

    @typing.override
    def __len__(self) -> int:
        return self._cumsum_len[-1]

    def append(self, td: TensorDict, /) -> None:
        self.tensordicts.append(td)

    def pop(self) -> TensorDict:
        return self.tensordicts.pop()

    @typing.override
    def _getitem(self, idx: IntArray, /) -> TensorDict:
        # Which tensordict to use in ``self.tensordicts``.
        td_idx: IntArray = np.searchsorted(self._cumsum_len, idx, side="right")
        assert td_idx.shape == idx.shape

        # How many elements are in the partitions prior to the current.
        prior_elements: IntArray = np.roll(self._cumsum_len, 1)
        prior_elements[0] = 0

        # Index in partition = original index - elements in prior partitions.
        idx_in_part: IntArray = idx - prior_elements[td_idx]

        # TensorDict that each index would correspond to.
        td_for_idx: list[TensorDict] = [self.tensordicts[t] for t in td_idx]

        assert len(idx_in_part) == len(td_for_idx)

        tds: list[TensorDict] = []
        for td, part_idx in zip(td_for_idx, idx_in_part):
            assert -len(td) <= part_idx < len(td), {
                "index for sub partition": part_idx,
                "tensordict's length": len(td),
            }
            tds.append(td[part_idx])
        return tensordict.stack(tds, dim=0)

    @property
    def _cumsum_len(self) -> IntArray:
        return np.cumsum([len(d) for d in self.tensordicts])


def is_list_of_tensordict(data) -> TypeIs[list[TensorDict]]:
    return isinstance(data, list) and all(isinstance(t, TensorDict) for t in data)
