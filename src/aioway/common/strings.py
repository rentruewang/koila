# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from . import functions


class Stringer(Protocol):
    @abc.abstractmethod
    def __call__(self) -> str: ...


class LazyStr(Stringer):
    """
    Evaluates to string on ``__call__`` or ``__str__`` or ``__repr__``.
    """

    def __init__(self, string: str | Stringer) -> None:
        """
        Args
            string: The string production factory or a plain ``str``.
        """

        if isinstance(string, str):
            self._string = lambda: string
        else:
            self._string = string

    def __call__(self) -> str:
        """
        Evaluates the underlying string.

        Returns:
            A string.
        """

        return self._string()

    def __str__(self) -> str:
        """
        Produce a string representation of the current object.

        Returns:
            The string representation.
        """

        return self()

    @functions.wraps(__str__)
    def __repr__(self) -> str:
        return str(self)
