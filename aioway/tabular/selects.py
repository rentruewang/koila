# Copyright (c) RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Callable

from tensordict import TensorDict

from aioway.errors import AiowayError

__all__ = ["SelectExec", "ProjectExec", "FilterColumnExec"]


@dcls.dataclass(frozen=True)
class SelectExec(ABC):
    def __call__(self, data: TensorDict) -> TensorDict:
        keys = self._keys(data)
        return data.select(*keys)

    @abc.abstractmethod
    def _keys(self, data: TensorDict, /) -> list[str]: ...


@dcls.dataclass(frozen=True)
class ProjectExec(SelectExec):
    target: list[str]

    def _keys(self, data: TensorDict) -> list[str]:
        if redundant := set(self.target).difference(data.keys()):
            raise SelectMissingKeyError(f"Redundant keys: {redundant} provided.")

        return self.target


@dcls.dataclass(frozen=True)
class FilterColumnExec(SelectExec):
    predicate: Callable[[str], bool]

    def _keys(self, data: TensorDict) -> list[str]:
        return [key for key in data.keys() if self.predicate(key)]


class SelectMissingKeyError(AiowayError, KeyError): ...
