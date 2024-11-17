# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol, TypeVar

from .nodes import Node
from .walkers import Walker

_T = TypeVar("_T", bound=Node, contravariant=True)
_E = TypeVar("_E", bound=Node, covariant=True)


class Rewriter(Walker[_T, _E], Protocol[_T, _E]):
    """
    ``Rewriter`` rewrites the given expression,
    acting like a translator or compiler between different language.s
    """

    @abc.abstractmethod
    def __call__(self, node: _T) -> _E: ...
