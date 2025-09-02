# Copyright (c) AIoWay Authors - All Rights Reserved

from typing import Protocol


class Compiler[I, O](Protocol):
    """
    Compiler transforms a ``Node`` representation into another.
    """

    def __call__(self, node: I) -> O: ...
