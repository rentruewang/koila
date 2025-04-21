# Copyright (c) RenChu Wang - All Rights Reserved

from collections.abc import Callable
from typing import Protocol

__all__ = ["Proc"]


class Proc[F: Callable](Protocol):
    """
    A protocol that defines a callable object that takes a function
    and return the same function with a different signature.
    """

    def __call__(self, func: F, /) -> F:
        """
        A callable that takes a function and returns a function.
        """

        ...
