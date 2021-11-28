from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass
class PartialInfo:
    index: slice | None
    total: int


class CallBack(Protocol):
    @abstractmethod
    def __call__(self, input: Tensor) -> Tensor:
        ...


def trivial(input: Tensor) -> Tensor:
    return input


def mean(ratio: float) -> CallBack:
    def callback(input: Tensor) -> Tensor:
        return input / ratio

    return callback
