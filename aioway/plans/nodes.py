# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from collections.abc import Sequence
from typing import ClassVar, Protocol

__all__ = ["Node"]


class Node[T: "Node"](Protocol):
    """
    ``Node`` is a protocol that represents a node in a graph.

    Todo:
        Make changes to ``Node`` such that it checks subclass's params,
        to simulate a scala-like annotation of ``T <: Node[T]``.

    Todo:
        ``Protocol``'s abstract ``property`` methods are causing issues for ``dataclass``es,
        because the ``dataclass`` decorator does not know how to overwrite ``property``s.
        Consider making all abstract functions functions rather than protocols,
        unless only used for ``typing.runtime_checkable``.
    """

    __match_args__: ClassVar[tuple[str, ...]]
    """
    ``Node`` should support structural decomposition to make pattern matching easier.
    """

    @property
    @abc.abstractmethod
    def sources(self) -> Sequence[T]:
        """
        The sources of a node must also be sources, or an empty ``Sequence``.

        See the documentation of ``T`` for more information.

        Returns:
            The sources node sequence.
        """

        ...
