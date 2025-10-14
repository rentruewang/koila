# Copyright (c) AIoWay Authors - All Rights Reserved

import abc
import logging
from abc import ABC
from collections.abc import Iterator
from typing import Self

from tensordict import TensorDict
from tensordict._td import TensorDict

__all__ = ["Exec"]


LOGGER = logging.getLogger(__name__)


class Exec(Iterator[TensorDict], ABC):
    """
    ``Exec`` is a ``BatchGen``, reponsible for managing runtime behavior.
    Responsible for launching an ``BatchGen`` everytime ``__iter__`` is called.

    It is responsible to execute a ``Thunk``,
    and has a 1 to 1 relationship with ``Thunk``s,
    where each ``Thunk`` would require an ``Exec`` to run.
    """

    @abc.abstractmethod
    def __iter__(self) -> Self:
        """
        The function ``iter(self)`` calls.
        Perform necessary operations to copy / fork the current generator.
        """

    @abc.abstractmethod
    def __next__(self) -> TensorDict:
        """
        Getting the next item.
        """
