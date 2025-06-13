# Copyright (c) AIoWay Authors - All Rights Reserved

from typing import Protocol

from aioway.nodes import Node


class Compiler[I: Node, O: Node](Protocol):
    """
    Compiler transforms a ``Node`` representation into another.
    """

    def __call__(self, node: I) -> O: ...
