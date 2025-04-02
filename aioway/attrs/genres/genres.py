# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from typing import Protocol

from aioway.attrs.sets import AttrSet

__all__ = ["AttrGenre"]


class AttrGenre(Protocol):
    def __contains__(self, obj: object) -> bool:
        if not isinstance(obj, AttrSet):
            return False

        return self.contains(obj)

    @abc.abstractmethod
    def contains(self, attrs: AttrSet, /) -> bool:
        """
        Compute whether or not the current definition of ``AttrGenre``
        encompasses the ``AttrSet`` definition.

        Args:
            attrs: The ``AttrSet`` instance to compare to.

        Returns:
            A boolean.
        """

        ...
