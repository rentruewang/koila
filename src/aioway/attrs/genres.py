# Copyright (c) AIoWay Authors - All Rights Reserved

__all__ = ["AttrGenre"]

import abc
from abc import ABC

from aioway._errors import AiowayError

from .attrs import AttrSet


class AttrGenre(ABC):
    def __contains__(self, item: object) -> bool:
        if isinstance(item, AttrSet):
            return self._accepts(item)

        raise AiowayGenreError(
            f"Item: {item} is not an `AttrSet`, does not know how to check."
        )

    @abc.abstractmethod
    def _accepts(self, attrs: AttrSet) -> bool:
        """
        Check if the attributes are accepted by this genre.
        """

        ...


class AiowayGenreError(AiowayError, TypeError): ...
