# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol, TypeVar

from .nodes import Node
from .walkers import Walker

__all__ = ["Rewriter"]

T = TypeVar("T", bound=Node, contravariant=True)
E = TypeVar("E", bound=Node, covariant=True)


class Rewriter(Walker[T, E], Protocol[T, E]):
    """
    ``Rewriter`` rewrites the given expression,
    acting like a translator or compiler between different language.s
    """

    @abc.abstractmethod
    def __call__(self, node: T) -> E: ...
