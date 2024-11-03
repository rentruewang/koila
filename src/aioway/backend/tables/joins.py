# Copyright (c) 2024 RenChu Wang - All Rights Reserved

import dataclasses as dcls
from typing import TypeVar

from aioway.backend.volatile import BinaryExec, Block

from .tables import Table

_T = TypeVar("_T")


@dcls.dataclass(frozen=True)
class JoinTable(Table):
    """
    ``JoinTable`` is a ``Table`` that depends on 2 ``Table``s.
    For example, products in relational algebra can be implemented using a ``JoinTable``.
    """

    executor: BinaryExec
    "The function that takes 2 ``Table``s as inputs."

    left: Table
    "The left ``Table`` operator."

    right: Table
    "The right ``Table`` operator."

    def __call__(self) -> Block:
        l = self.left()
        r = self.right()
        output = self.executor(l, r)
        return output

    def accept(self, visitor: Table.Visitor[_T]) -> _T:
        return visitor.join(self)

    @property
    def sources(self) -> tuple[Table, Table]:
        return self.left, self.right
