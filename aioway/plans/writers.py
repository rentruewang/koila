# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from .nodes import Node
from .walkers import Walker

__all__ = ["Rewriter"]


class Rewriter[T: Node, E: Node](Walker, Protocol):
    """
    ``Rewriter`` rewrites the given expression,
    acting like a translator or compiler between different language.s
    """

    @abc.abstractmethod
    def __call__(self, node: T) -> E: ...
