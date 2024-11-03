# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from collections.abc import Generator, Iterable, Iterator
from typing import TypeVar

from .nodes import Node

_T = TypeVar("_T", bound=Node)


@dcls.dataclass(frozen=True)
class Tree(Iterable[Node[_T]]):
    root: Node[_T]
    """
    The root node for the current tree.
    """

    def __iter__(self) -> Iterator[Node[_T]]:
        """
        Post order traversal of the current tree.
        """

        yield from _post_order(self.root)


def _post_order(root: Node[_T]) -> Generator[Node[_T], None, None]:
    for child in root.sources:
        yield from _post_order(child)

    yield root
