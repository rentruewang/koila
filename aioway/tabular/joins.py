# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections import defaultdict as DefaultDict
from collections.abc import Hashable
from typing import LiteralString

import numpy as np
import torch
from numpy.typing import NDArray
from tensordict import TensorDict

from aioway.errors import AiowayError

from .batches import BatchList

__all__ = ["JoinExec", "MergeJoinExec", "HashJoinExec"]


@dcls.dataclass(frozen=True)
class JoinExec(ABC):
    on: str
    """
    The column on which to match.
    """

    def __call__(self, left: BatchList, right: BatchList, /) -> BatchList:
        self._check(left, "lhs")
        self._check(right, "rhs")

        return self._join(left=left, right=right)

    @abc.abstractmethod
    def _join(self, left: BatchList, right: BatchList) -> BatchList: ...

    def _check(self, data: TensorDict, side: LiteralString) -> None:
        side = side.capitalize()

        if self.on not in data:
            raise MissingJoinKeyError(f"{side} does not contain key {self.on}")

        if not len(data):
            raise UnbatchedError(f"{side} is not batched and cannot be indexed.")


@dcls.dataclass(frozen=True)
class MergeJoinExec(JoinExec):
    def _join(self, left: BatchList, right: BatchList) -> BatchList:
        lhs: DefaultDict[Hashable, list[int]] = DefaultDict(list)
        for idx, key in enumerate(left[self.on]):
            lhs[key.item()].append(idx)

        rhs: DefaultDict[Hashable, list[int]] = DefaultDict(list)
        for idx, key in enumerate(right[self.on]):
            rhs[key.item()].append(idx)

        common_keys = {*lhs.keys()} & {*rhs.keys()}

        left_keys = []
        right_keys = []

        for k in common_keys:
            left_keys = lhs[k]
            right_idx = rhs[k]

            left_cnt = len(left_keys)
            right_cnt = len(right_idx)

            # FIXME: Use np.repeat.
            left_keys.extend(np.repeat(left_keys, right_cnt).tolist())
            right_keys.extend(
                sum(([right_idx[i]] * left_cnt for i in range(right_cnt)), [])
            )

        return np.array([left_keys, right_keys])


@dcls.dataclass(frozen=True)
class HashJoinExec(JoinExec):
    def _join(self, left: TensorDict, right: TensorDict) -> NDArray:
        left_col = left[self.on]
        right_col = right[self.on]

        product = left_col[:, None] == right_col[None, :]
        indices = torch.nonzero(product)
        return indices.cpu().numpy()


class MissingJoinKeyError(AiowayError, KeyError): ...


class UnbatchedError(AiowayError, IndexError): ...
