# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from abc import ABC
from collections.abc import Iterator

from aioway.blocks import Block

__all__ = ["Stream"]


class Stream(ABC):
    """
    ``Stream`` represents a possibly unbound flow of data produced by a source,
    it is one of the main physical abstractions in ``aioway``.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Block]:
        """
        Iterator is the main API of a ``Stream``, where the input is batched and produced.

        Yields:
            The blocks coming in streams.
        """

        ...
