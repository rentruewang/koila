# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from typing import Generic, Iterator, Mapping, NamedTuple, TypeVar

from aioway.plans import Node

from .previews import Preview
from .ranges import Range

__all__ = ["ConcreteSet", "Configurable", "FreeSet", "ParamSet"]

T = TypeVar("T")

type Primitives = str | int | float | bool


@dcls.dataclass(frozen=True)
class ParamSet(Mapping[str, T]):
    params: dict[str, T]

    def __iter__(self) -> Iterator[str]:
        return iter(self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, key: str) -> T:
        return self.params[key]


class Resolution(NamedTuple):
    unresolved: "FreeSet"
    resolved: "ConcreteSet"


@dcls.dataclass(frozen=True)
class FreeSet(ParamSet[Range]):
    def __call__(self, **kwargs: Primitives) -> Resolution:
        raise NotImplementedError


@dcls.dataclass(frozen=True)
class ConcreteSet(Mapping[str, str | int | float | bool]):
    pass


@dcls.dataclass(frozen=True)
class Configurable(Node["Configurable[T]"], Generic[T], ABC):
    @abc.abstractmethod
    def __call__(self, params: FreeSet) -> Preview[T]: ...

    def parameters(self) -> FreeSet:
        return FreeSet(params=self._parameters())

    @abc.abstractmethod
    def _parameters(self) -> dict[str, Range]: ...
