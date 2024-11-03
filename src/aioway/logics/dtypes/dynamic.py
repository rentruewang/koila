# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import abc
import typing
from typing import Protocol, TypeVar

_T = TypeVar("_T", covariant=True)


@typing.runtime_checkable
class DynamicType(Protocol[_T]):
    """
    ``DynamicType`` is a protocol that carries its typing information at runtime.
    Due to Python's typing information being discarded at runtime,
    and the fact that a compiler cannot know the types of each node beforehand,
    this protocol would be used to store the type information of each node.
    """

    @property
    @abc.abstractmethod
    def dtype(self) -> type[_T]:
        """
        The dynamic type of the node.
        """

        ...
