# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import dataclasses as dcls
from abc import ABC
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from aioway.logics import Node, Relation

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class Info(Generic[_T]):
    relation: type[Relation]
    """
    The relation for which this preview module is for.
    """

    shape: tuple[int, ...]
    """
    The shape of the current module
    """

    parameters: int
    """
    The number of parameters for a given function.
    """

    initialization: Callable[[], _T]


@dcls.dataclass(frozen=True)
class Preview(Node["Preview[_T]"], Generic[_T], ABC):
    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Info[_T]: ...
