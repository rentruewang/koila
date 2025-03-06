# Copyright (c) RenChu Wang - All Rights Reserved

import abc
from collections.abc import Iterator
from typing import Protocol

__all__ = ["Sampler"]


class Sampler(Protocol):
    """
    The sampler protocol.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[int]: ...
