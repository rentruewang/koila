# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol, TypeVar

from .nodes import Node
from .walkers import Walker

_T = TypeVar("_T", bound=Node, contravariant=True)
_E = TypeVar("_E", bound=Node, covariant=True)


class Rewriter(Walker[_T, _E], Protocol[_T, _E]):
    @abc.abstractmethod
    def __call__(self, node: _T) -> _E: ...
